import torch, math 
import torch.nn.functional as F 

BLOCK_SIZE = 1024 
NEG_INF = -1e10

N = 1024 
d = 256 

def flash_attention(Q, K, V):
    O = torch.zeros_like(Q, requires_grad=True)
    l = torch.zeros(N)
    m = torch.ones(N) * NEG_INF

    Bc = BLOCK_SIZE 
    Br = min(BLOCK_SIZE, d)

    Q_blocks = torch.split(Q, Br)
    K_blocks = torch.split(K, Br) 
    V_blocks = torch.split(V, Br)

    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    O_blocks = list(torch.split(O, Br))
    l_blocks = list(torch.split(l, Br))
    m_blocks = list(torch.split(m, Br))

    for j in range(Tc):
        Kj, Vj = K_blocks[j], V_blocks[j]
        for i in range(Tr):
            Qi = Q_blocks[i]
            Oi = O_blocks[i]
            li = l_blocks[i]
            mi = m_blocks[i]

            Sij = Qi @ Kj.T # (Br, Bc)

            mij, _ = torch.max(Sij, dim=-1, keepdim=True)
            Pij = torch.exp(Sij - mij)
            lij = torch.sum(Sij, dim=-1, keepdim=True)
            Pij_Vj = Pij @ Vj 

            mi_new = torch.maximum(mij, mi)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij - mi_new) * lij 

            O_blocks[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (torch.exp(mij - mi_new) / li_new) * Pij_Vj 
            l_blocks[i] = li_new 
            m_blocks[i] = mi_new 

    O = torch.cat(O_blocks)
    return O 

def normal_attention(Q, K, V):
    QKt = Q @ K.T 
    attn = F.softmax(QKt, dim=-1)
    return attn @ V

if __name__=="__main__":
    R = torch.randn(N, 3 * d)
    Q, K, V = torch.split(R, 256, dim=-1) 

    flash_O = flash_attention(Q, K, V) 
    normal_O = normal_attention(Q, K, V) 

    print(torch.isclose(flash_O, normal_O).all())
