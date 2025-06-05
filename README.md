# Mestrado

Este reposit\xC3\xB3rio cont\xC3\xA9m exemplos para conectar ao Amazon Athena usando Python e a biblioteca boto3.

Veja o arquivo `connect_athena.py` para um script simples de conex\xC3\xA3o e execu\xC3\xA7\xC3\xA3o de consultas.


## Instalação

Instale as dependências listadas em `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Configura\xC3\xA7\xC3\xA3o de credenciais

As credenciais da AWS devem ser definidas no arquivo `aws_config.py`. Substitua os
valores de `AWS_ACCESS_KEY_ID` e `AWS_SECRET_ACCESS_KEY` pelas suas credenciais
reais **apenas localmente**. N\xC3\xA3o envie credenciais verdadeiras para o
reposit\xC3\xB3rio.

Exemplo:

```python
AWS_ACCESS_KEY_ID = "SUA_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY = "SEU_SECRET_KEY"
AWS_REGION = "us-east-1"
```

Em seguida, execute `connect_athena.py` para testar a conex\xC3\xA3o.
