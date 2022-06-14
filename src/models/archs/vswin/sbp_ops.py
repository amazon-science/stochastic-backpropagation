import torch


def assert_grad_mask(grad_mask, downsample):
    """make sure the grad_mask and downsample rate is matching.

    Args:
        grad_mask (torch.Tensor): any shape.
        downsample (int): downsample rate.

    """
    if grad_mask is not None:
        assert (
            downsample == grad_mask.numel() / grad_mask.sum()
        ), f"downsample and grad_mask must match. downsample:{downsample}, grad_mask_downsample: {grad_mask.numel() / grad_mask.sum()}"


class SBPmlp(object):
    # todo: merge this class into SBPMlpFunc.
    @staticmethod
    def forward(grad_mask, downsample, forward_fn, x, *params):
        """
        Args:
          grad_mask (torch.Tensor[B,D]): torch.bool. True means kept and False means dropt.
          downsample (int): The number to downsample along the temporal dimension.
          forward_fn (callable): The forward function in MLP.
          x (torch.Tensor[B,D,H,W,C]): input to MLP.
          params (tuple): The learnable parameters in MLP.

        Returns: tuple.
          (output, tensors_to_save).
          - output: shape [B,D,H,W,C]. The output of mlp.
          - tensors_to_save (tuple): The tensors needed to be saved for backward.
        """
        B, D, H, W, C = x.shape
        Dd = D // downsample

        # forward
        with torch.no_grad():
            y, random_tensor = forward_fn(x)

        # save for backward
        x_w_grad = x.masked_select(grad_mask.view(B, D, 1, 1, 1)).view(B, Dd, H, W, C)
        tensors_to_save = (
            forward_fn,
            x_w_grad,
            params,
            random_tensor,
        )

        return y, tensors_to_save

    @staticmethod
    def backward(saved_tensors, dy):
        """
        Args:
          dy (torch.Tensor[B,Dd,H,W,C]): gradient of output. zeros are removed.
        """

        (
            forward_fn,
            x_w_grad,
            params,
            random_tensor,
        ) = saved_tensors

        # detach x_w_grad
        x_w_grad = x_w_grad.detach().requires_grad_(True)

        with torch.enable_grad():
            y, random_tensor = forward_fn(x_w_grad, random_tensor)
        input_grads = torch.autograd.grad(y, (x_w_grad,) + params, dy)
        return (None, None, None) + input_grads


class SBPMlpFunc(torch.autograd.Function):
    """Mlp Function with stochastic backpropagation"""

    @staticmethod
    def forward(ctx, grad_mask, downsample, forward_fn, x, *params):
        ctx.grad_mask = grad_mask
        ctx.downsample = downsample
        assert_grad_mask(grad_mask, downsample)
        y, tensors_to_save = SBPmlp.forward(
            grad_mask, downsample, forward_fn, x, *params
        )
        ctx.tensors = tensors_to_save
        return y

    @staticmethod
    def backward(ctx, dy):
        """
        Args:
          dy (torch.Tensor[B,D,H,W,C]): gradient of output. zeros are removed.
        """
        B, D, H, W, C = dy.shape
        grad_mask = ctx.grad_mask
        Dd = D // ctx.downsample
        dy = dy.masked_select(grad_mask.view(B, D, 1, 1, 1)).view(B, Dd, H, W, C)

        grads = list(SBPmlp.backward(ctx.tensors, dy))
        dx = torch.zeros(B, D, H, W, C, dtype=dy.dtype, device=dy.device)
        dx.masked_scatter_(grad_mask.view(B, D, 1, 1, 1), grads[3])
        grads[3] = dx
        del ctx.tensors, ctx.grad_mask, ctx.downsample
        return tuple(grads)


class PFDotProductAttention_v4(torch.autograd.Function):
    """
    y = softmax(q @ k.T + bias) @ v

    drop q, A but not k, v
    """

    @staticmethod
    def run_fn(q, k, v, bias, scale, mask):
        Nq = q.shape[2]
        B_, nH, Nk, C = k.shape

        q = q * scale
        A = q @ k.transpose(-2, -1) + bias

        if mask is not None:
            nW = mask.shape[0]
            A = A.view(B_ // nW, nW, nH, Nq, Nk) + mask.unsqueeze(1).unsqueeze(0)
            A = A.view(B_, nH, Nq, Nk)

        A = torch.softmax(A, dim=-1)  # shape: [B,nH,Nq,Nk]
        y = A @ v
        return y

    @staticmethod
    def forward(ctx, q, k, v, bias, qk_scale, mask=None, grad_mask=None, downsample=1):
        """forward m-a

        Args:
            ctx (context object): context object.
            q (torch.Tensor[B_,nH,N,C]): query. N = window_size. B_ = B * num_windows
            k (torch.Tensor[B_,nH,N,C]): key.
            v (torch.Tensor[B_,nH,N,C]): value.
            bias (torch.Tensor[nH,N,N]): position bias.
            mask (torch.Tensor[num_windows, N, N]): Attention mask.
            grad_mask (torch.Tensor[B_,N]): torch.bool. True means kept and False means dropt.
            downsample (int): The number to downsample along the temporal dimension.

        Returns: torch.Tensor[B_,nH,N,C]. attention.

        """
        assert_grad_mask(grad_mask, downsample)
        B_, nH, N, C = q.shape
        scale = qk_scale
        with torch.no_grad():
            y = PFDotProductAttention_v4.run_fn(q, k, v, bias, scale, mask)

        if downsample == 1:
            ctx.tensors = (q, k, v, bias, scale, mask)
        else:
            Nd = N // downsample
            indices = torch.arange(0, N, device=q.device)
            indices = indices.masked_select(grad_mask[0]).tolist()
            q = q[:, :, indices, :].contiguous()  # [B_, nH, Nd, C]
            bias = bias[:, indices, :].contiguous()  # [nH, Nd, N]
            if mask is not None:
                # mask = mask.permute(1, 0, 2)[indices].permute(1, 0, 2).contiguous()
                mask = mask[:, indices, :].contiguous()
                # mask = mask.masked_select(grad_mask[:, None, :, None]).view(nH, Nd, N)

            ctx.tensors = (q, k, v, bias, scale, mask)
            ctx.indices = indices
        ctx.grad_mask = grad_mask
        ctx.downsample = downsample
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        """
        Args:
            dy (torch.Tensor[B_,nH,N,C])
        """

        # shape:
        # Nd = N //downsample
        # A: [B_,nH,Nd,N]
        # q: [B_,nH,Nd,C]
        # k, v: [B_,nH,N,C]
        downsample = ctx.downsample
        grad_mask = ctx.grad_mask
        q, k, v, bias, scale, mask = ctx.tensors
        B_, nH, Nd, C = q.shape
        N = Nd * downsample

        if downsample != 1:
            indices = ctx.indices
            # dy = dy.permute(2, 0, 1, 3)[indices].permute(1, 2, 0, 3).contiguous()
            dy = dy.masked_select(grad_mask[:, None, :, None]).view(B_, nH, Nd, C)

        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            bias = bias.detach().requires_grad_(True)
            y = PFDotProductAttention_v4.run_fn(q, k, v, bias, scale, mask)
        dq, dk, dv, d_bias = torch.autograd.grad(y, (q, k, v, bias), dy)

        # upsample back
        if downsample != 1:
            B_, nH, N, C = k.shape
            dq = torch.zeros(
                B_, nH, N, C, dtype=dq.dtype, device=dq.device
            ).masked_scatter_(grad_mask[:, None, :, None], dq)

            d_bias1 = torch.zeros(N, nH, N, dtype=d_bias.dtype, device=d_bias.device)
            d_bias1[indices] = d_bias.permute(1, 0, 2)
            d_bias1 = d_bias1.permute(1, 0, 2).contiguous()
            d_bias = d_bias1

            # d_bias1 = torch.zeros(
            #     B_, nH, N, N, dtype=d_bias.dtype, device=d_bias.device
            # )
            # d_bias1.masked_scatter_(grad_mask[:, None, :, None], d_bias)
            # d_bias1 = d_bias1.sum(0)
            # d_bias = d_bias1

        del ctx.tensors, ctx.grad_mask
        return dq, dk, dv, d_bias, None, None, None, None
