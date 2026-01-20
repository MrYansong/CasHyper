import torch
import torch.nn as nn
from typing import Tuple
from config.Constants import DEVICE

def _make_sedenion_sign_table() -> torch.Tensor:
    table_idx = [
        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15],
        [ 1, 0, 3, 2, 5, 4, 7, 6, 9, 8,11,10,13,12,15,14],
        [ 2, 3, 0, 1, 6, 7, 4, 5,10,11, 8, 9,14,15,12,13],
        [ 3, 2, 1, 0, 7, 6, 5, 4,11,10, 9, 8,15,14,13,12],
        [ 4, 5, 6, 7, 0, 1, 2, 3,12,13,14,15, 8, 9,10,11],
        [ 5, 4, 7, 6, 1, 0, 3, 2,13,12,15,14, 9, 8,11,10],
        [ 6, 7, 4, 5, 2, 3, 0, 1,14,15,12,13,10,11, 8, 9],
        [ 7, 6, 5, 4, 3, 2, 1, 0,15,14,13,12,11,10, 9, 8],
        [ 8, 9,10,11,12,13,14,15, 0, 1, 2, 3, 4, 5, 6, 7],
        [ 9, 8,11,10,13,12,15,14, 1, 0, 3, 2, 5, 4, 7, 6],
        [10,11, 8, 9,14,15,12,13, 2, 3, 0, 1, 6, 7, 4, 5],
        [11,10, 9, 8,15,14,13,12, 3, 2, 1, 0, 7, 6, 5, 4],
        [12,13,14,15, 8, 9,10,11, 4, 5, 6, 7, 0, 1, 2, 3],
        [13,12,15,14, 9, 8,11,10, 5, 4, 7, 6, 1, 0, 3, 2],
        [14,15,12,13,10,11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1],
        [15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ]
    sign_raw = [
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
        [ 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1],
        [ 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1],
        [ 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1],
        [ 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1],
        [ 1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1],
        [ 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1],
        [ 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1],
        [ 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1],
        [ 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1],
        [ 1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1],
        [ 1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1],
        [ 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1],
        [ 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
    ]
    table_idx = torch.tensor(table_idx, dtype=torch.long)
    sign = torch.tensor(sign_raw, dtype=torch.float32)
    return table_idx, sign

_IDX_TABLE, _SIGN_TABLE = _make_sedenion_sign_table()

class Sedenion:
    def __init__(self, tensor: torch.Tensor):
        if tensor.shape[-2] != 16:
            raise ValueError("Second-to-last dimension must be 16 for sedenion [e0..e15].")
        self.tensor = tensor

    def __add__(self, other: "Sedenion") -> "Sedenion":
        return Sedenion(self.tensor + other.tensor)

    def multiply(self, other: "Sedenion") -> "Sedenion":
        if not isinstance(other, Sedenion):
            raise TypeError("Inputs must be Sedenion instances")
        a = [self.tensor[..., k, :] for k in range(16)]
        b = [other.tensor[..., k, :] for k in range(16)]

        res = [torch.zeros_like(a[0]) for _ in range(16)]

        for i in range(16):
            for j in range(16):
                k = _IDX_TABLE[i, j].item()
                sign = _SIGN_TABLE[i, j].item()
                res[k] += sign * a[i] * b[j]

        result = torch.stack(res, dim=-2)
        return Sedenion(result)

    def score_multiply(self, other: "Sedenion") -> torch.Tensor:

        if not isinstance(other, Sedenion):
            raise TypeError("Inputs must be Sedenion instances")
        a = self.tensor[..., 0, :]   # [..., D]
        b = other.tensor[..., 0, :]
        real = a * b
        for i in range(1, 16):
            real -= self.tensor[..., i, :] * other.tensor[..., i, :]
        return real

    def conjugate(self) -> "Sedenion":
        conj = self.tensor.clone()
        conj[..., 1:, :] = -conj[..., 1:, :]
        return Sedenion(conj)

    def norm(self) -> torch.Tensor:
        return torch.sqrt(torch.sum(self.tensor ** 2, dim=(-2, -1)))

    def inverse(self) -> "Sedenion":
        conj = self.conjugate().tensor
        norm_sq = torch.sum(self.tensor ** 2, dim=(-2, -1), keepdim=True)[..., None]
        inv_tensor = conj / (norm_sq + 1e-10)
        return Sedenion(inv_tensor)
    def __str__(self) -> str:
        return f"Sedenion(tensor_shape={tuple(self.tensor.shape)})"

class Quaternion:
    def __init__(self, tensor):
        if tensor.shape[-2] != 4:
            raise ValueError("Second-to-last dimension must be 4 for quaternion [a, b, c, d].")
        self.tensor = tensor

    def __add__(self, other):

        return Quaternion(self.tensor + other.tensor)

    def multiply(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError("Inputs must be Quaternion instances")

        a1, b1, c1, d1 = [self.tensor[..., i, :] for i in range(4)]
        a2, b2, c2, d2 = [other.tensor[..., i, :] for i in range(4)]
        real = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        i = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        j = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        k = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

        result = torch.stack([real, i, j, k], dim=-2)
        return Quaternion(result)
    
    def score_multiply(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError("Inputs must be Quaternion instances")
        a1, b1, c1, d1 = [self.tensor[..., i, :] for i in range(4)]
        a2, b2, c2, d2 = [other.tensor[..., i, :] for i in range(4)]
        real = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        return real

    def conjugate(self):
        conj = self.tensor.clone()
        conj[..., 1:, :] = -conj[..., 1:, :]
        return Quaternion(conj)

    def norm(self):
        return torch.sqrt(torch.sum(self.tensor ** 2, dim=(-2, -1)))

    def inverse(self):
        conj = self.conjugate().tensor
        norm_sq = torch.sum(self.tensor ** 2, dim=(-2, -1), keepdim=True)[..., None]
        inverse = conj / (norm_sq + 1e-10)
        return Quaternion(inverse)

    def __str__(self):
        return f"Quaternion(tensor_shape={self.tensor.shape})"


class Octonion:
    def __init__(self, tensor):
        if tensor.shape[-2] != 8:
            raise ValueError("Second-to-last dimension must be 8 for octonion [e0, e1, e2, e3, e4, e5, e6, e7].")
        self.tensor = tensor

    def __add__(self, other):
        return Octonion(self.tensor + other.tensor)

    def multiply(self, other):
        if not isinstance(other, Octonion):
            raise TypeError("Inputs must be Octonion instances")
        e0, e1, e2, e3, e4, e5, e6, e7 = [self.tensor[..., i, :] for i in range(8)]
        f0, f1, f2, f3, f4, f5, f6, f7 = [other.tensor[..., i, :] for i in range(8)]

        r0 = e0 * f0 - e1 * f1 - e2 * f2 - e3 * f3 - e4 * f4 - e5 * f5 - e6 * f6 - e7 * f7
        r1 = e0 * f1 + e1 * f0 + e2 * f4 - e3 * f7 + e4 * f2 + e5 * f6 - e6 * f5 - e7 * f3
        r2 = e0 * f2 - e1 * f4 + e2 * f0 + e3 * f6 - e4 * f1 - e5 * f7 + e6 * f3 + e7 * f5
        r3 = e0 * f3 + e1 * f7 - e2 * f6 + e3 * f0 + e4 * f5 - e5 * f4 - e6 * f2 + e7 * f1
        r4 = e0 * f4 - e1 * f2 - e2 * f1 - e3 * f5 + e4 * f0 + e5 * f3 - e6 * f7 + e7 * f6
        r5 = e0 * f5 - e1 * f6 + e2 * f7 - e3 * f4 - e4 * f3 + e5 * f0 + e6 * f1 + e7 * f2
        r6 = e0 * f6 + e1 * f5 - e2 * f3 + e3 * f2 + e4 * f7 - e5 * f1 + e6 * f0 - e7 * f4
        r7 = e0 * f7 + e1 * f3 - e2 * f5 + e3 * f1 - e4 * f6 - e5 * f2 + e6 * f4 + e7 * f0

        result = torch.stack([r0, r1, r2, r3, r4, r5, r6, r7], dim=-2)
        return Octonion(result)

    def score_multiply(self, other):
        e0, e1, e2, e3, e4, e5, e6, e7 = [self.tensor[..., i, :] for i in range(8)]
        f0, f1, f2, f3, f4, f5, f6, f7 = [other.tensor[..., i, :] for i in range(8)]

        r0 = e0 * f0 - e1 * f1 - e2 * f2 - e3 * f3 - e4 * f4 - e5 * f5 - e6 * f6 - e7 * f7
        return r0
    
    def conjugate(self):
        conj = self.tensor.clone()
        conj[..., 1:, :] = -conj[..., 1:, :]
        return Octonion(conj)

    def norm(self):
        return torch.sqrt(torch.sum(self.tensor ** 2, dim=(-2, -1)))

    def inverse(self):
        conj = self.conjugate().tensor
        norm_sq = torch.sum(self.tensor ** 2, dim=(-2, -1), keepdim=True)[..., None]
        inverse = conj / (norm_sq + 1e-10)
        return Octonion(inverse)

    def __str__(self):
        return f"Octonion(tensor_shape={self.tensor.shape})"

class Complex:
    def __init__(self, tensor):
        if tensor.shape[-2] != 2:
            raise ValueError("Second-to-last dimension must be 2 for complex [a, b].")
        self.tensor = tensor

    def __add__(self, other):
        return Complex(self.tensor + other.tensor)

    def multiply(self, other):
        if not isinstance(other, Complex):
            raise TypeError("Inputs must be Complex instances")
        a1, b1 = self.tensor[..., 0, :], self.tensor[..., 1, :]
        a2, b2 = other.tensor[..., 0, :], other.tensor[..., 1, :]
        real = a1 * a2 - b1 * b2
        imag = a1 * b2 + b1 * a2

        result = torch.stack([real, imag], dim=-2)
        return Complex(result)
    
    def score_multiply(self, other):
        if not isinstance(other, Complex):
            raise TypeError("Inputs must be Complex instances")
        a1, b1 = self.tensor[..., 0, :], self.tensor[..., 1, :]
        a2, b2 = other.tensor[..., 0, :], other.tensor[..., 1, :]
        real = a1 * a2 - b1 * b2
        return real

    def conjugate(self):
        conj = self.tensor.clone()
        conj[..., 1, :] = -conj[..., 1, :]
        return Complex(conj)

    def norm(self):
        return torch.sqrt(torch.sum(self.tensor ** 2, dim=(-2, -1)))

    def inverse(self):
        conj = self.conjugate().tensor
        norm_sq = torch.sum(self.tensor ** 2, dim=(-2, -1), keepdim=True)[..., None]
        inverse = conj / (norm_sq + 1e-10)
        return Complex(inverse)

    def __str__(self):
        return f"Complex(tensor_shape={self.tensor.shape})"


class hypercomplex_Linear(nn.Module):
    def __init__(self, input_dim, hyper_dim, output_dim, concat = True, dropout_ratio=0.1):
        super().__init__()
        self.in_channels = input_dim
        self.hyper_dim = hyper_dim
        self.out_channels = output_dim
        self.concat = concat
        self.dropout_ratio = dropout_ratio
        if self.concat:
            self.linear = nn.Linear(self.hyper_dim * self.in_channels, self.hyper_dim * self.out_channels, bias=True)
            self.layer_morm = nn.LayerNorm(normalized_shape=self.hyper_dim * self.out_channels)
        else:
            self.w = nn.Parameter(torch.FloatTensor(self.hyper_dim, self.in_channels))
            self.b = nn.Parameter(torch.FloatTensor(self.hyper_dim, self.in_channels))
            nn.init.xavier_normal_(self.w)
            nn.init.xavier_normal_(self.b)
            if self.hyper_dim == 4:
                self.w1 = Quaternion(self.w)
                self.b1 = Quaternion(self.b)
            elif self.hyper_dim == 2:
                self.w1 = Complex(self.w)
                self.b1 = Complex(self.b)
            elif self.hyper_dim == 8:
                self.w1 = Octonion(self.w)
                self.b1 = Octonion(self.b)
            elif self.hyper_dim == 16:
                self.w1 = Sedenion(self.w)
                self.b1 = Sedenion(self.b)
            self.layer_morm = nn.LayerNorm(normalized_shape=self.out_channels)
        
        self.dropout = nn.Dropout(self.dropout_ratio)
        
        self.Lin = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, input):
        original_shape = input.shape
        if self.concat:
            if len(input.shape) >= 3:
                assert input.shape[-2] == self.hyper_dim, 'Quaternion_Linear: input shape [-2] is not matched.'
                input = torch.flatten(input, start_dim=-2, end_dim=-1)
            else:
                assert input.shape[-1] == self.in_channels * self.hyper_dim, 'Quaternion_Linear: input shape [-1] is not matched.'
            new_shape = original_shape[:-2] + (self.hyper_dim, self.out_channels)
            result = self.linear(input)
            result = self.dropout(result)
            result = result.reshape(new_shape)
        else:
            if self.hyper_dim == 4:
                if not isinstance(input, Quaternion):
                    input = Quaternion(input)
            elif self.hyper_dim == 8:
                if not isinstance(input, Octonion):
                    input = Octonion(input)
            elif self.hyper_dim == 2:
                if not isinstance(input, Complex):
                    input = Complex(input)
            elif self.hyper_dim == 16:
                if not isinstance(input, Complex):
                    input = Sedenion(input)

            result = input.multiply(self.w1) + self.b1
            result = self.Lin(result.tensor)
            result = self.dropout(result)
        return result
