# Vector Store using Langchain
This is a repo to chat with data stored in a given set of documents using a local LLM. This repo uses langchain for prompt writing and api calls, huggingface via a langchain module for document embedding, ChromaDB via a langchain module for storing document embeddings, and text-generation-webui as the api, i.e., the LLM with which the user interacts. 

Documents to be stored in the vector store (m1.txt, m2.txt, etc) were copy/pasted from wikipedia and are about mushrooms. 

## text-generation-webui
To get an LLM and an api set up locally on your computer, see:
https://github.com/oobabooga/text-generation-webui

You will have to pick/download a model to use, quantized models are recommended for personal computers. 

When starting text-generation-webui include the --api flag, for example:

```
.\start_windows.bat --api
```

This will enable an api endpoint at localhost:5000 which is used in the langchain code.  

## Run
After setting up an endpoint, in the terminal, run:

```
python bot2.py
```

This will split, embed and store the vectorized documents. It will also send the given question in the code (line 58) using a pre-made langchain prompt, stuffed with relevant document splits from the vector store to the LLM via the api. Langchain verbosity is set to true, thus the full prompt will be output to terminal. It is possible to save the vector store to disk so it doesn't have to be embedded/stored every time the code is run. 

## requirements.txt
A list of non-standard requirements for running this repo. Some of these packages are under rapid development and getting everything to work may require some troubleshooting. Some known issues are listed below.  

### Issue: TextGen module
Recently the langchain textgen module, which is supposed to mimic an openAI endpoint, stopped working with text-generation-webui.
As a workaround, an old version of the module is copy/pasted and directly imported as opposed to being imported from langchain, see:
https://github.com/langchain-ai/langchain/issues/14318

### Issue: Windows install
Windows does not have the required package pwd. See below for a work around. It entails changing some python package code. 
https://github.com/langchain-ai/langchain/issues/17514

## Discussion
This is a first iteration of this repo. A docker version should be feasible and can avoid the Windows issues. This is a work in progress. Any issues/questions contact adam.conovaloff@analytica.net. 
