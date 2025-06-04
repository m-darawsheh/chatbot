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
1. fist type of module is the pretrained text generating module like (deepseek, gpt2)
  * this type of module take a text and try to complete it (he has a lot of words in his brain and he will use it to complete you sentense)
  * if you give him a Q he will try to complete it not answring for example if a asked (what is today) the answer will be (what is today's news?"
  "I'm not sure," said the man. "I'm not sure what's going on. I'm not sure what's going on. I'm not sure what's going on. I'm not sure what's going) 
  because the module is not for answering qustions but it has like a small brain that good at completing the sentanse 
  * to fix this i can fine-tuning the module (future traing) on a data that i has so now the module can take a Q and answer the same way but now he the answer will be a bit better because i fine-tuning it on a data so he will use some of this data to answer 
  so he will stay the same only completing the text but now he will use some my data to complete from it. 
  * and there a another way to make the module better is but prompt engineering which is leting him completing the sentase as alawys
  but now i will let him completing the answer not the qustion and i will resturcter the prompt in a way that let the module completing the answer in a good, way so i will need to get the aswar first and add it to the prompt and let the module complete it 
  example the quastion is (who are you) i will let it look like this (the Q is : who are you
  the aswner is : i am mohammed 
  Q : who am i
  AI answer : 
  )
  and he will complete it 
  * also you can do both ways (prompt engineering + fine-tuning) and it is the requmanded.
2. secoend type of module is the QA(qustion answering) module.
  * the kind of module also called Retrieval (because it is aculy not answering it is just giving you the closist thing to it in the contaxt) 
  * in this type of AI module you have to give the qustion and the contax which the module will look for the answer for that qustion the that contaxt and if the answer is not in the cotaxt the module will not answer the qustion 
  * but the return will be only the pice of words that answer the qustion example the q is what is your name and the cotaxt is hello my name is mohammed the return will be mohammed
  * there are another type of QA module this kind of QA module will that your sentases and return them to vectors and then you job is to find the similarty between these vectors
  * in short there are to type of QA first one you give it the contetx and the Q and it return the answer from the context and the other type of QA module he take your sentenses and return them to vector and also return your Q to vector and you have to find the closest vector form the sentanse to the Q using cosine_similarty 
3. so in our sitiwation we have a data and we what the chatBot to answer to acording to that data  
  * first we need a module to find that relevant snippet (peice of data) in our data
  * now after getting the relevant data now we want to change it from just row data to be a good looking answer so we will us a module and some prompt engineering to do so 
  * to find the relevant snippet we will use (all-MiniLM-L6-v2) module and after the prompt engineering we will use (microsoft/DialoGPT-medium) for making the relevant snippet looks more like an answer.


note : what if we can't use more than one module and is module is a text generating module what to do now 
first we will manualy look for the answer in the contaxt and use it for prompt engineering and also we can train the module in our data.