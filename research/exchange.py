"""片側のArUco検知を契機とする、要求→応答→確認の遭遇共有。

受信側も同じID/地図/試行/条件を検証する。送信成功と相手の受信確認を区別。
研究データを周期的にブロードキャストしない。
"""
import time
import uuid
from research.world import fingerprint


class EncounterExchange:
    def __init__(self, world, network, mode='full', cooldown=8, log=None):
        self.world, self.network = world,network
        self.setup_hash = fingerprint(world.setup)
        self.mode, self.cooldown = mode,cooldown
        self.log = log or (lambda *args,**kwargs:None)
        self.pending = {}
        self.responses = {}
        self.response_retries = {}
        self.finished = {}
        self.last_seen = {}
        self.cooldowns = {}

    def envelope(self, peer, kind, transaction, body=None):
        return dict(protocol=1,kind=kind,transaction=transaction,source=self.world.robot_id,
                    target=peer,map_id=self.world.map_id,trial_id=self.world.trial_id,
                    mode=self.mode,setup_hash=self.setup_hash,body=body)

    def seen(self, peer, marker_id, now=None):
        now = time.monotonic() if now is None else now
        if peer==self.world.robot_id or peer not in self.world.setup['robots']:
            return
        self.last_seen[peer] = now
        if self.mode=='none' or peer in self.pending or now < self.cooldowns.get(peer,0):
            return
        self.cooldowns[peer] = now+self.cooldown
        tx = uuid.uuid4().hex
        msg = self.envelope(peer,'request',tx,self.world.snapshot(self.mode))
        msg['marker_id'] = marker_id
        self.pending[peer] = dict(tx=tx,message=msg,created=now,last=now,attempts=1)
        self.network.send(peer,msg)
        self.log('exchange_requested',peer=peer,transaction=tx)

    def receive(self, peer, message, now=None):
        now = time.monotonic() if now is None else now
        try:
            if self.mode=='none':
                return
            expected = dict(protocol=1,source=peer,target=self.world.robot_id,map_id=self.world.map_id,
                            trial_id=self.world.trial_id,mode=self.mode,setup_hash=self.setup_hash)
            if peer not in self.world.setup['robots'] or any(message.get(k)!=v for k,v in expected.items()):
                raise ValueError('地図・試行・機体ID・共有条件が不一致')
            tx = message.get('transaction')
            if not isinstance(tx,str) or len(tx)!=32:
                raise ValueError('transaction不正')
            kind = message.get('kind')
            key = (peer,tx)
            if kind=='request':
                if message.get('marker_id')!=self.world.setup['robots'][self.world.robot_id]['marker_id']:
                    raise ValueError('検知マーカーIDと宛先が不一致')
                if key not in self.responses:
                    self.world.merge(message['body'])
                    reply = self.envelope(peer,'response',tx,self.world.snapshot(self.mode))
                    self.responses[key] = (now,reply)
                    self.response_retries[key] = [now,1]
                    self.log('exchange_received',peer=peer,transaction=tx,cubes=len(self.world.cubes()))
                self.network.send(peer,self.responses[key][1])
            elif kind=='response':
                pending = self.pending.get(peer)
                if pending and pending['tx']==tx and now-pending['created']<=20:
                    self.world.merge(message['body'])
                    del self.pending[peer]
                    self.finished[key] = now
                    self.network.send(peer,self.envelope(peer,'ack',tx))
                    self.log('exchange_success',peer=peer,transaction=tx,cubes=len(self.world.cubes()))
                elif key in self.finished:
                    self.network.send(peer,self.envelope(peer,'ack',tx))
            elif kind=='ack' and key in self.responses:
                if key not in self.finished:
                    self.finished[key] = now
                    self.log('exchange_success',peer=peer,transaction=tx,cubes=len(self.world.cubes()))
            else:
                return
        except (ValueError,KeyError,TypeError) as error:
            self.log('exchange_rejected',peer=peer,reason=str(error))

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        for peer,msg,received in self.network.receive():
            if now-received<=20:
                self.receive(peer,msg,now)
        for peer,p in list(self.pending.items()):
            if now-p['created']>20:
                self.log('exchange_timeout',peer=peer,transaction=p['tx'])
                del self.pending[peer]
            elif now-p['last']>=3 and p['attempts']<5 and now-self.last_seen.get(peer,-100)<2:
                self.network.send(peer,p['message'])
                p['last'],p['attempts'] = now,p['attempts']+1
        for key,(created,reply) in self.responses.items():
            if key in self.finished:
                continue
            retry = self.response_retries[key]
            if now-created>20:
                if retry[1]!=-1:
                    self.log('exchange_ack_timeout',peer=key[0],transaction=key[1])
                    retry[1] = -1
            elif now-retry[0]>=3 and retry[1]<5:
                self.network.send(key[0],reply)
                retry[:] = [now,retry[1]+1]
        self.responses = {k:v for k,v in self.responses.items() if now-v[0]<60}
        self.response_retries = {k:v for k,v in self.response_retries.items() if k in self.responses}
        self.finished = {k:v for k,v in self.finished.items() if now-v<60}
