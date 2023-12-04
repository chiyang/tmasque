from torch import nn
import torch
import numpy as np

import os

class Encoder(nn.Module):
  def __init__(self, latent_dims, qc_level):
    super(Encoder, self).__init__()
    dims = []
    if qc_level == 1:
      dims = [17, 24, 8, latent_dims]
    elif qc_level == 2:
      dims = [22, 36, 12, latent_dims]
    elif qc_level == 3:
      dims = [8, 12, latent_dims]
    if qc_level == 3:
      self.linear1  = None
      self.linear2  = nn.Linear(dims[0], dims[1])
      self.linear2_bn = nn.BatchNorm1d(dims[1])
      self.linear3A = nn.Linear(dims[1], dims[2])
      self.linear3B = nn.Linear(dims[1], dims[2])
    else:
      self.linear1  = nn.Linear(dims[0], dims[1])
      self.linear1_bn = nn.BatchNorm1d(dims[1])
      self.linear2  = nn.Linear(dims[1], dims[2])
      self.linear2_bn = nn.BatchNorm1d(dims[2])
      self.linear3A = nn.Linear(dims[2], dims[3])
      self.linear3B = nn.Linear(dims[2], dims[3])
  def forward(self, x):
    if self.linear1 is not None:
      x = torch.tanh(self.linear1(x))
      x = torch.tanh(self.linear1_bn(x))
    x = torch.tanh(self.linear2(x))
    x = torch.tanh(self.linear2_bn(x))
    mu = self.linear3A(x)
    logvar = self.linear3B(x)
    return mu, logvar


class QualityEncoder(object):
  def __init__(self, device='auto', encoder_type1_path=None, encoder_type2_path=None, encoder_type3_path=None, encoder_dim = [2, 2, 2]):
    self.encoder_type1_path = encoder_type1_path
    self.encoder_type2_path = encoder_type2_path
    self.encoder_type3_path = encoder_type3_path
    if not self.encoder_type1_path:
      self.encoder_type1_path = os.path.join(os.path.split(__file__)[0], 'encoder', 'quality_encoder_type1.pickle')
    if not self.encoder_type2_path:
      self.encoder_type2_path = os.path.join(os.path.split(__file__)[0], 'encoder', 'quality_encoder_type2.pickle')
    if not self.encoder_type3_path:
      self.encoder_type3_path = os.path.join(os.path.split(__file__)[0], 'encoder', 'quality_encoder_type3.pickle')
    self.type1_encoder = Encoder(encoder_dim[0], qc_level=1)
    self.type2_encoder = Encoder(encoder_dim[1], qc_level=2)
    self.type3_encoder = Encoder(encoder_dim[2], qc_level=3)
    self.type1_quality_refs = [
      [0.0,  1.0, 0.0, 0.0, 0.050, 0.250, 0.0,    0.0,  1.0, 0.0, 0.0, 0.050, 0.250, 0.0,   1.0, 0.0, 0.0],
      [0.3,  0.4, 0.3, 0.3, 0.335, 0.475, 0.3,    0.3,  0.4, 0.3, 0.3, 0.335, 0.475, 0.3,   0.7, 0.3, 0.3],
      [1.0, -1.0, 1.0, 1.0, 1.000, 1.000, 5.0,    0.0,  1.0, 0.0, 0.0, 0.050, 0.250, 0.0,   0.0, 1.0, 1.0],
      [0.0,  1.0, 0.0, 0.0, 0.050, 0.250, 0.0,    1.0, -1.0, 1.0, 1.0, 1.000, 1.000, 5.0,   0.0, 1.0, 1.0],
      [1.0, -1.0, 1.0, 1.0, 1.000, 1.000, 5.0,    1.0, -1.0, 1.0, 1.0, 1.000, 1.000, 5.0,   0.0, 1.0, 1.0]
    ]
    self.type2_quality_refs = [
      [0.0,  1.0,  1.0, 0.0, 0.0, 0.200, 0.200,   0.0,  1.0,  1.0, 0.0, 0.0, 0.200, 0.200,   1.0, 0.0,  1.0,  1.0, 0.0, 0.0, 0.200, 0.200],
      [0.2,  0.6,  0.6, 0.2, 0.2, 0.360, 0.360,   0.2,  0.6,  0.6, 0.2, 0.2, 0.360, 0.360,   0.6, 0.2,  0.6,  0.6, 0.2, 0.2, 0.360, 0.360],
      [1.0, -1.0, -1.0, 1.0, 1.0, 1.000, 1.000,   0.0,  1.0,  1.0, 0.0, 0.0, 0.050, 0.250,  -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.000, 1.000], # good heavy only; worst others
      [0.0,  1.0,  1.0, 0.0, 0.0, 0.050, 0.250,   1.0, -1.0, -1.0, 1.0, 1.0, 1.000, 1.000,  -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.000, 1.000], # bad heavy only
      [1.0, -1.0, -1.0, 1.0, 1.0, 1.000, 1.000,   1.0, -1.0, -1.0, 1.0, 1.0, 1.000, 1.000,  -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.000, 1.000]
    ]
    self.type3_quality_refs = [
      [0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.5,  0.3,  0.9, 1.5, 0.3, 0.9, 0.3, 0.3],
      [5.0,  1.0,  3.0, 0.0, 0.0, 0.0, 1.0, 1.0],
      [0.0,  0.0,  0.0, 5.0, 1.0, 3.0, 1.0, 1.0],
      [5.0,  1.0,  3.0, 5.0, 1.0, 3.0, 1.0, 1.0]
    ]
    if device == 'auto':
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device
    print('device: ' + self.device)
    self.load_encoder()
    self._update_reference_latents()
  
  def __call__(self, quality_vectors, normalize=True):
    qv = torch.tensor(quality_vectors, dtype=torch.float32).to(self.device)
    dim = qv.shape[1]
    if dim == 17:
      latent = self.type1_encoder(qv)[0].cpu().detach().numpy()
      score = self.score_type1_latent(latent)
      type = 1
    elif dim == 22:
      latent = self.type2_encoder(qv)[0].cpu().detach().numpy()
      score = self.score_type2_latent(latent)
      type = 2
    elif dim == 8:
      latent = self.type3_encoder(qv)[0].cpu().detach().numpy()
      score = self.score_type3_latent(latent)
      type = 3
    else:
      raise 'Unkonw dimension. Valid dimensions are 17 for type 1 quality, 22 for type 2 quality, and 8 for type 3 quality.'
    if normalize:
      return self.normalize_score(type, score), latent
    else:
      return score, latent

  def load_encoder(self):
    self.type1_encoder.load_state_dict(torch.load(self.encoder_type1_path, map_location=self.device))
    self.type2_encoder.load_state_dict(torch.load(self.encoder_type2_path, map_location=self.device))
    self.type3_encoder.load_state_dict(torch.load(self.encoder_type3_path, map_location=self.device))
    self.type1_encoder.to(self.device)
    self.type2_encoder.to(self.device)
    self.type3_encoder.to(self.device)
    self.type1_encoder.eval()
    self.type2_encoder.eval()
    self.type3_encoder.eval()
    self._update_reference_latents()

  def _update_reference_latents(self):
    self.type1_latent_points = self.type1_encoder(torch.tensor(self.type1_quality_refs).to(self.device))[0].to('cpu').detach().numpy()
    self.type2_latent_points = self.type2_encoder(torch.tensor(self.type2_quality_refs).to(self.device))[0].to('cpu').detach().numpy()
    self.type3_latent_points = self.type3_encoder(torch.tensor(self.type3_quality_refs).to(self.device))[0].to('cpu').detach().numpy()
    self._score_range = [
      dict(max= self.score_type1_latent([self.type1_latent_points[0]])[0], min=self.score_type1_latent([self.type1_latent_points[4]])[0]),
      dict(max= self.score_type2_latent([self.type2_latent_points[0]])[0], min=self.score_type2_latent([self.type2_latent_points[4]])[0]),
      dict(max= self.score_type3_latent([self.type3_latent_points[0]])[0], min=self.score_type3_latent([self.type3_latent_points[4]])[0])
    ]
  def encode_quality(self, quality_vectors):
    qv = torch.tensor(quality_vectors, dtype=torch.float32).to(self.device)
    dim = qv.shape[1]
    if dim == 17:
      return self.type1_encoder(torch.tensor(qv, dtype=torch.float32).to(self.device))[0].detach().numpy()
    elif dim == 22:
      return self.type2_encoder(torch.tensor(qv, dtype=torch.float32).to(self.device))[0].detach().numpy()
    elif dim == 8:
      return self.type3_encoder(torch.tensor(qv, dtype=torch.float32).to(self.device))[0].detach().numpy()
    else:
      raise 'Unkonw dimension. Valid dimensions are 17 for type 1 quality, 22 for type 2 quality, and 8 for type 3 quality.'

  # def encode_type1_quality(self, quality_vector):
  #   return self.type1_encoder(torch.tensor(quality_vector, dtype=torch.float32).to(self.device))[0].detach().numpy()
  # def encode_type2_quality(self, quality_vector):
  #   return self.type2_encoder(torch.tensor(quality_vector, dtype=torch.float32).to(self.device))[0].detach().numpy()
  # def encode_type3_quality(self, quality_vector):
  #   return self.type3_encoder(torch.tensor(quality_vector, dtype=torch.float32).to(self.device))[0].detach().numpy()

  def score_type1_latent(self, type1_latents):
    return self.score_func(type1_latents, self.type1_latent_points)
  def score_type2_latent(self, type2_latents):
    return self.score_func(type2_latents, self.type2_latent_points)
  def score_type3_latent(self, type3_latents):
    return self.score_func(type3_latents, self.type3_latent_points)

  def normalize_score(self, type, score):
    if type <= 0 or type >=4:
      return
    min = self._score_range[type - 1]['min']
    max = self._score_range[type - 1]['max']
    return -10 + 20 * ((score - min)/(max - min))

  def score_func(self, latent_points, ref_latent_points):
    # ref_latent_points = self._ref_latent_points
    dist_max = np.linalg.norm(ref_latent_points[0] - ref_latent_points[4])
    score = 2 * (1 - (self.dist_func(latent_points, ref_latent_points[0])/dist_max)**0.5)
    score = score + 1 * (1 - (self.dist_func(latent_points, ref_latent_points[1])/dist_max)**0.5)
    score = score + 1 * (1 - (self.dist_func(latent_points, ref_latent_points[2])/dist_max)**0.5)
    score = score - 1 * (1 - (self.dist_func(latent_points, ref_latent_points[3])/dist_max)**0.5)
    score = score - 2 * (1 - (self.dist_func(latent_points, ref_latent_points[4])/dist_max)**0.5)
    return score

  def dist_func(self, a, b):
    return np.linalg.norm(a - b, axis=1)
