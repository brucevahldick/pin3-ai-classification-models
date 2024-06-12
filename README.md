# pin3-ai-classification-models

### Script para a construção do modelo A
- O artefato correspondente ao script para a construção do modelo A é o script_modelo_a.py dentro de pin3-backend
- Argumentos para utilização do script
  - evaluate: avalia o modelo com o conjunto de dados de validção
  - train: treina o modelo com o conjunto de dados de treino e avalia ele com o conjunto de dados de validação
  - retrain: 

### Script para a construção do modelo B
- O artefato correspondente ao script para a construção do modelo B é o script_modelo_b.py dentro de pin3-backend
- Argumentos para utilização do script
  - evaluate: avalia o modelo com o conjunto de dados de validção
  - train: treina o modelo com o conjunto de dados de treino e avalia ele com o conjunto de dados de validação
  - retrain: 

### Script para teste e comparação dos modelos
- O script para fazer a comparação das métricas de ambos os modelos é o compare_models.py dentro de pin3-backend
- Rodar o arquivo vai fazer uma bateria de testes com os dados separados para teste no dataset e apresentará acurácia, recall e precisão de ambos os modelos

### Aplicação Web
- Iniciar o app.py dentro de pin3-backend que corresponde ao servidor do flask
- Rodar o comando "npm run dev" no pin3-frontend e abrir o link do servidor para acessar o frontend