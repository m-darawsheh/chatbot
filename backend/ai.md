* now we have an python backend server that can take a qustoin and give you an answer 
* now we need to use the AI to get a good answer but how to
    - first way to use an AI is to use the openai API (we ask the openAI servers and they give as an answer)
      - we can ask them by sending the http request using requset lib (like the fetch in js). 
      - or we can use the openia lib which will make things esaier (we do not have to use the request lib to took with openai servers now we can took to them useing openia lib which is esaier)
      - in short you send a reqest for the openAI servers they run for you the leatest module (chat-gpt4) and then give you an answer but the (api's are limited) unless you pay.
    - but there are aonther way to use, AI you can istall the model in your computer, but who to use it !! in the first way we let the openAI servers to use it for us and give as the answer
      - but for sure i want tools to help me install that model and run it
      - after installing and run it now we can use it in our code and that's it 
      - then if we can install it and use it why do i need to us the openAI API's?
        - yes now we can use it and there is no need for the api but the model wont be as good as the API model and the istalled model will be slow in the CPU and will take alot of space
* now after we knew how to use the AI ethier using api or installing the model, we will go with the model aproch because we do not have mony($$).
    - to do that there are a lot of ways and tools to dnowlod and use AI's module but we will use a Hugging Face and Transformers 
    - what is Hugging Face ? Hugging Face is a company and open-source community that builds tools to make working with AI and machine learning easy and accessible
        (It’s like the GitHub of AI models.) 
    - transformers is a Python library made by Hugging Face that gives you easy access to thousands of pretrained language models.
    - in short Hugging Face is a place that contain alot of module and tools to help you use these module and in python if you want to use a hugging face module you will use a transformers lib (which also made up from hugging face) 
* now we will use transformers to install the module and use it.
    - every module need a tokenizer to work with it (because the module only deal with number so you need the tokenizr to chage between text and numbers).
    - now i need to install the module and load it so i can use it  (loading it mean that i get it from where i installed and then but it inside my python code so i can use it)
      to do so we will use the (transformers.AutoTokenizer.from_pretrained("gpt2")) to install and load the token and this (transformers.AutoModelForCausalLM.from_pretrained("gpt2"))
      to install and load the module 
    - now i have the module intalled (in the .cache) and loaded in my code 
    - so what is if i called the same two line of code again it will check if the module is in (.cache) and if so it will only load it and if not it will install it and load it. 
    - and you can change where does the HF installing the modules by changing the this env var (HF_HOME) 
* after i installed the module and knew how to use it and how to get the result but there are 2 type of modules  
1.   