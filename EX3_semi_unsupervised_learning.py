import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
import joblib # For saving the SVM model

# ==========================================
# 0. Configuration, Hyperparameters & Seed
# ==========================================
# Fix seed for reproducibility as required
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
INPUT_DIM = 784      # Fashion MNIST: 28x28
HIDDEN_DIM = 400
LATENT_DIM = 20
LABEL_COUNTS = [100, 600, 1000, 3000] # The different amounts of labels to test

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Data Preparation (Fashion MNIST with Balanced Sampling)
# ==========================================
def get_data_loaders(num_labeled):
    """
    Creates DataLoaders for FashionMNIST.
    - unlabeled_loader: All training data (for VAE).
    - labeled_loader: A balanced subset of 'num_labeled' samples (for SVM).
    - test_loader: The full test set.
    """
    transform = transforms.ToTensor()
    
    # Load FashionMNIST
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    # 1. Unlabeled Loader (Uses ALL training data)
    unlabeled_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Labeled Loader (Balanced Subset)
    # Requirement: "make sure you have an equal amount of examples from each class"
    targets = train_dataset.targets.numpy()
    labeled_indices = []
    samples_per_class = num_labeled // 10
    
    print(f"Creating balanced subset with {samples_per_class} samples per class...")
    for i in range(10): # For each of the 10 classes
        # Find indices of all images belonging to this class
        class_indices = np.where(targets == i)[0]
        # Randomly select the required number of samples without replacement
        # The fixed seed ensures this selection is consistent.
        selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        labeled_indices.extend(selected_indices)
        
    labeled_subset = Subset(train_dataset, labeled_indices)
    labeled_loader = DataLoader(labeled_subset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Test Loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return unlabeled_loader, labeled_loader, test_loader

# ==========================================
# 2. VAE Model Definition (M1 Feature Extractor)
# ==========================================
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc21 = nn.Linear(HIDDEN_DIM, LATENT_DIM) # Mean
        self.fc22 = nn.Linear(HIDDEN_DIM, LATENT_DIM) # LogVariance
        
        # Decoder
        self.fc3 = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.fc4 = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, INPUT_DIM))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss: Reconstruction + KL Divergence
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, INPUT_DIM), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ==========================================
# 3. Training & Extraction Logic
# ==========================================
def train_vae(model, loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(loader.dataset)

def extract_features(model, loader):
    """
    Passes data through the trained VAE encoder to get 'mu'.
    Returns numpy arrays for SVM.
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            # Encode to get latent representation (mu)
            mu, _ = model.encode(data.view(-1, INPUT_DIM))
            
            # Move to CPU and convert to numpy
            features.append(mu.cpu().numpy())
            labels.append(target.numpy())
            
    if len(features) > 0:
        return np.concatenate(features), np.concatenate(labels)
    else:
        return np.array([]), np.array([])

# ==========================================
# 4. Main Execution Loop
# ==========================================
if __name__ == "__main__":
    # --- Initial Data Load ---
    print("Preparing Fashion MNIST Data...")
    # Get the full unlabeled and test loaders. The labeled size here doesn't matter yet.
    unlabeled_loader, _, test_loader = get_data_loaders(num_labeled=100)
    print(f"Data Loaded: {len(unlabeled_loader.dataset)} Unlabeled samples for VAE training.")

    # --- Stage 1: Train VAE (Done once on all data) ---
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    print("\n--- STAGE 1: Training VAE (Unsupervised) ---")
    for epoch in range(1, EPOCHS + 1):
        loss = train_vae(vae, unlabeled_loader, optimizer)
        print(f"Epoch {epoch}: VAE Loss = {loss:.4f}")
    
    # Save VAE weights as required
    torch.save(vae.state_dict(), 'vae_fashion_mnist.pth')
    print("VAE weights saved to 'vae_fashion_mnist.pth'.")

    # Extract test features once, as they are the same for all runs
    print("Extracting features from the test set...")
    X_test, y_test = extract_features(vae, test_loader)

    # --- Loop through different labeled set sizes ---
    results = {}
    for num_labels in LABEL_COUNTS:
        print(f"\n=== Running experiment with {num_labels} labeled samples ===")
        
        # 1. Get balanced labeled data loader
        _, labeled_loader, _ = get_data_loaders(num_labeled=num_labels)
        
        # 2. Extract features for the labeled training set
        print(f"Extracting features for {num_labels} training samples...")
        X_train, y_train = extract_features(vae, labeled_loader)
        
        # 3. Train SVM
        # Requirement: "SVM (with a kernel of your choice)". We choose RBF.
        print(f"Training SVM (RBF Kernel) with {num_labels} samples...")
        clf = SVC(kernel='rbf', gamma='scale', C=1.0)
        clf.fit(X_train, y_train)
        
        # Save the SVM model as required
        svm_filename = f'svm_model_{num_labels}.pkl'
        joblib.dump(clf, svm_filename)
        print(f"SVM model saved to '{svm_filename}'.")

        # 4. Evaluate on test set
        print("Evaluating...")
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[num_labels] = acc
        print(f"Accuracy with {num_labels} labels: {acc * 100:.2f}%")

    # --- Final Summary ---
    print("\n=== Final Results on Fashion MNIST ===")
    print("Labels | Accuracy")
    print("-------|---------")
    for num_labels in LABEL_COUNTS:
        print(f"{num_labels:<7}| {results[num_labels] * 100:.2f}%")