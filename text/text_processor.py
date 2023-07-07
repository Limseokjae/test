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
            "content":f"{texts} \n  나는 위의 글에서 요리에 필요한 재료와 그 재료의 수량을 표로 만들려고 해. 항목은 ""재료"", ""수량""이고, 표의 항목은 탭으로 구분된 형태로 만들어줘. 만약 수량이 없으면 -로 만들어줘나는 위의 글에서 요리에 필요한 재료와 그 재료의 수량을 표로 만들려고 해. 항목은 ""재료"", ""수량""이고, 표의 항목은 구분자없이 탭으로 구분된 형태로 만들어줘. 만약 수량이 없으면 -로 만들어줘"
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

