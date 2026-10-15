"""Yield native decoder outputs as soon as each latent has been decoded.

Keep full-chunk normalization and conv2 identical; move only temporal concat /
pointwise unpatchify outside the native decoder loop. Requires exact RGB parity.
"""
import hashlib,inspect


class NativePixelParts:
    def __init__(self,wrapper):
        self.wrapper=wrapper;self.model=wrapper.model
        function=self.model.cached_decode.__func__
        self.unpatchify=function.__globals__['unpatchify']
        source=inspect.getsource(function)
        for anchor in ['x = self.conv2(z)','is_first_chunk = self._feat_map[0] is None','self._conv_idx = [0]']:
            assert anchor in source,anchor
        self.contract={'native_cached_decode_sha256':hashlib.sha256(source.encode()).hexdigest(),
                       'normalization_and_full_chunk_conv2_retained':True,
                       'native_per_latent_decoder_and_cache_retained':True,
                       'declared_change':'per-latent unpatchify/readback instead of waiting for full temporal concatenation',
                       'new_weights_or_interpolation':False}

    def pixels(self,latent):
        import torch
        assert latent.ndim==5 and latent.shape[0]==1
        wrapper=self.wrapper;model=self.model;device=latent.device;dtype=latent.dtype
        scale=[wrapper.mean.to(device=device,dtype=dtype),1.0/wrapper.std.to(device=device,dtype=dtype)]
        zs=latent.permute(0,2,1,3,4)
        with torch.autocast(device_type=device.type,dtype=dtype):
            z=zs[0].unsqueeze(0)
            z=z/scale[1].view(1,model.z_dim,1,1,1)+scale[0].view(1,model.z_dim,1,1,1)
            x=model.conv2(z);first=model._feat_map[0] is None
            for index in range(z.shape[2]):
                model._conv_idx=[0]
                before=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True);before.record()
                if index==0 and first:
                    decoded=model.decoder(x[:,:,index:index+1,:,:],feat_cache=model._feat_map,feat_idx=model._conv_idx,first_chunk=True)
                else:
                    decoded=model.decoder(x[:,:,index:index+1,:,:],feat_cache=model._feat_map,feat_idx=model._conv_idx)
                decoded=self.unpatchify(decoded,patch_size=2);after.record()
                pixels=decoded.float().clamp_(-1,1).squeeze(0)
                pixels=torch.stack([pixels],dim=0).permute(0,2,1,3,4)
                yield pixels,{'latent_offset':index,'native_gpu_begin':before,'native_gpu_end':after}
