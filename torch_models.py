import torch
import torch.nn as nn

# Multilayer perceptron
class MLP(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x))
        return self.layers(x)


##############
# FusionModel
##############

# Encoders for feature families

# Encoder for scalars
class EncoderScalar(nn.Module):
    def __init__(self, n_features=1, embedding_dim=32):
        super().__init__()
        self.proj = nn.Linear(n_features, embedding_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.proj(x))

# Encoder for non-temporal families with > 1 features
class EncoderNonTemporal(nn.Module):
    def __init__(self, n_features, hidden_threshold=4, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.use_hidden = n_features > hidden_threshold

        if self.use_hidden:
            # For larger sets: hidden layer + output
            self.fc1 = nn.Linear(n_features, hidden_dim)
            self.act1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, embedding_dim)
            self.act2 = nn.ReLU()
        else:
            # For small sets: single linear layer
            self.proj = nn.Linear(n_features, embedding_dim)
            self.act = nn.ReLU()

    def forward(self, x):
        if self.use_hidden:
            x = self.act1(self.fc1(x))
            x = self.act2(self.fc2(x))
            return x
        else:
            return self.act(self.proj(x))

# Encoder for temporal families
class EncoderTemporal(nn.Module):
    def __init__(self, input_dim=1, embedding_dim=32, hidden_dim=32):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out, h_n = self.rnn(x)
        embedding = self.act(self.proj(h_n[-1]))
        return embedding

def get_encoders(families_indices, embedding_dims):
    """
    Get encoders for the chosen families and indices.
    embedding_dim for each encoder depends on type of
    family, e.g temporal, non-temporal, scalar
    """

    encoders = {} # store encoder for each family
    families_embedding_dims = {} # store embedding_dim for each family
    tot_dimension = 0
    for fam_name, indices in families_indices:

        # Select embedding dimension
        fam = feature_families[fam_name]
        if fam.temporal:
            embedding_dim = embedding_dims['temporal']
        elif fam.n_features == 1:
            embedding_dim = embedding_dims['scalar']
        elif fam.n_features > 1:
            embedding_dim = embedding_dims['non_temporal']

        families_embedding_dims[fam_name] = embedding_dim
        tot_dimension += embedding_dim

        # Select encoder
        n_indices = len(indices)
        if n_indices == 1:
            encoder = EncoderScalar(n_features=1, embedding_dim=embedding_dim)
        elif fam.temporal:
            encoder = EncoderTemporal(input_dim=1, hidden_dim=32, embedding_dim=embedding_dim)
        else:
            encoder = EncoderNonTemporal(n_features=n_indices, embedding_dim=embedding_dim)

        encoders[fam_name] = encoder
    return encoders, families_embedding_dims, tot_dimension

# FusionModel
class FusionModel(nn.Module):
    def __init__(self, encoders, input_dim, output_dim=1):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        embeddings = []
        for fam_name, encoder in self.encoders.items():
            embeddings.append(encoder(x[fam_name]))
        combined = torch.cat(embeddings, dim=1)
        return torch.sigmoid(self.mlp(combined))
