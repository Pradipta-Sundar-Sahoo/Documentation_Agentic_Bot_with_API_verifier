- [level 0](http://54.90.127.180:8501/)
- [level 1](http://54.90.127.180:8502/)
- Level 2 is made , BUT 3 Tmux session in a single instance was not running ... You can Set the API keys in local and Run it 

```
OPENAI_API_KEY=""
SERPER_API_KEY=""
LLAMA_CLOUD_API_KEY=""
```

```
streamlit run level2.py
```



## level0
Simple RAG (memory wagera hei)

  
## level1 : level0 + Used Agentic Framework 
- Supervisior Node
- Tool Node( 2 tools --> 1) google search, for external queries 2) query_vectordb --> for crustdataAPI usage queries
- Validator Node ( which validates API)


## level2:
 level1+ added two things url and pdf uploader .... where if you upload it will update in db



## Major Developements really required 

- Caches ( ye redis/memcache se ho jayega , TTL wagera set karke , Redis LEGEND)
- Database versioning ( may be you want the prev vectordb)
- Main Koi NoSQL wagera chahiye , to store data. (mongo is ok)
- Agentic framework ---> more tools and more better
- URL parser is not perfect ... some websites it cannot scan (((But PDF one is best, Because LLAMA PARSE is ❤️))) ---- url waale pe kaam karna hei
- Have to use Qdrant ( rust backend hai , so will be faster for sure)
- level0 pe simple RAG ya isme bhi agentic 
- Evaluation is must
  



NOTES:

- Geninuely , I had to make it in hurry(5hrs){deployment mein issue thha} due to some previous commitments , please consider if you like.
- I could make it to Slack BOT easily , but really some previous commitments till 15th jan
- Please inform me if this pleases you ,  becuase AWS ec2 cost and API cost .. i need to turn it off.
- May have issue with prompt template , could be fixed next turn.




Great Problem Statement.. Love it ❤️
