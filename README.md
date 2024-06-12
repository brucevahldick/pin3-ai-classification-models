# pin3-ai-classification-models

### Script para a construção do modelo A
- O artefato correspondente ao script para a construção do modelo A é o script_modelo_a.py dentro de pin3-backend
- Argumentos para utilização do script
  - evaluate: avalia o modelo com o conjunto de dados de validção
  - train: treina o modelo com o conjunto de dados de treino e avalia ele com o conjunto de dados de validação
  - retrain: retreina o modelo permitindo que sejam passados parametros difernetes a fim de se encontrar um modelo mais otimizado

Exemplo de utilização
```shell
-- To evaluate the model
python .\script_modelo_a.py evaluate

-- To train an first version of model
python .\script_modelo_a.py train

-- To retrain the model and try to find an optimized version
python .\script_modelo_a.py retrain 3 0.9 5 0.01 130
```

### Script para a construção do modelo B
- O artefato correspondente ao script para a construção do modelo B é o script_modelo_b.py dentro de pin3-backend
- Argumentos para utilização do script
  - evaluate: avalia o modelo com o conjunto de dados de validção
  - train: treina o modelo com o conjunto de dados de treino e avalia ele com o conjunto de dados de validação
  - retrain: 

Exemplo de utilização
```shell
-- To evaluate the model
python .\script_modelo_b.py evaluate

-- To train an first version of model
python .\script_modelo_b.py train

-- To retrain the model and try to find an optimized version
python .\script_modelo_b.py retrain 4 0.02 0.9
```

### Script para teste e comparação dos modelos
- O script para fazer a comparação das métricas de ambos os modelos é o compare_models.py dentro de pin3-backend
- Rodar o arquivo vai fazer uma bateria de testes com os dados separados para teste no dataset e apresentará acurácia, recall e precisão de ambos os modelos
```shell
-- To compare both models
python .\compare_models.py
```


### Aplicação Web
- Iniciar o app.py dentro de pin3-backend que corresponde ao servidor do flask
- Rodar o comando "npm run dev" no pin3-frontend e abrir o link do servidor para acessar o frontend