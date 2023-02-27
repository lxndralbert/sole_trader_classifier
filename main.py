import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the customer data
df = pd.read_csv('customer_data.csv')

# Preprocess the data
le = LabelEncoder()
df['business_indicators'] = le.fit_transform(df['business_indicators'])
ohe = OneHotEncoder()
contextual_info_ohe = ohe.fit_transform(df['contextual_info'].values.reshape(-1, 1)).toarray()

# Create the neural network model
class SoleTraderClassifier(nn.Module):
    def __init__(self, num_features, num_contextual_info_categories):
        super(SoleTraderClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8 + num_contextual_info_categories, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, c):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        c = torch.tensor(c, dtype=torch.float32)
        x = torch.cat((x, c), dim=1)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Train the model
num_features = 2
num_contextual_info_categories = contextual_info_ohe.shape[1]
model = SoleTraderClassifier(num_features, num_contextual_info_categories)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for epoch in range(1000):
    inputs = torch.tensor(df[['transaction_history', 'business_indicators']].values, dtype=torch.float32)
    labels = torch.tensor(df['is_sole_trader'].values, dtype=torch.float32).unsqueeze(1)
    contextual_info = contextual_info_ohe.astype(float)
    outputs = model(inputs, contextual_info)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the model
with torch.no_grad():
    inputs = torch.tensor(df[['transaction_history', 'business_indicators']].values, dtype=torch.float32)
    labels = torch.tensor(df['is_sole_trader'].values, dtype=torch.float32).unsqueeze(1)
    contextual_info = contextual_info_ohe.astype(float)
    outputs = model(inputs, contextual_info)
    predicted_labels = (outputs >= 0.5).squeeze().numpy()
    accuracy = (predicted_labels == df['is_sole_trader'].values).mean()

print('Accuracy: {:.2f}%'.format(accuracy * 100))