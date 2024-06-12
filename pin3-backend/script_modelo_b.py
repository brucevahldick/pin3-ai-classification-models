import os
from pathlib import Path

# Certifique-se de que o caminho para os dados está correto
data_path = Path('path_to_your_data')
model_path_fastai = 'model_fastai.pkl'
model_path_pytorch = 'model_pytorch.pth'

# Importar funções previamente definidas

from pyt import train_and_save_model

# Execução Principal
if __name__ == '__main__':
    # Treinar e salvar o modelo FastAI
    train_and_save_model(20)

