import openai

def get_info(keys,texts):
    """
        Args :
            keys (str) : openai api key
            texts (str,list) : text contents

        Returns :
            response (str) : chat gpt response
    """

    ## key info required
    openai.api_key = keys

    ## prompt 
    msg= [{"role": "user",
            "content":f"{texts} \n 여기서 재료에 대해서 [재료,수량] 형식으로  요약해줘"
        }]

    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=msg
    )
    response = completion.choices[0].message.content
    

    ## save
    # with open('./gpttest/tmp.txt','a') as f:
    #     f.write(response)

    return response

