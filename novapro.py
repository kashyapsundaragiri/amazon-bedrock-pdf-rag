import boto3
import json

prompt_data = """
Act as a researcher and provide a detailed analysis of the following topic: "The impact of artificial intelligence on modern healthcare systems." Please include recent advancements, potential benefits, ethical considerations, and future implications. Use credible sources and provide citations where applicable.
"""

bedrock = boto3.client(service_name="bedrock-runtime", region_name="ap-southeast-2")

payload = {
    "messages": [
        {
            "role": "user",
            "content": [{"text": prompt_data}]
        }
    ],
    "inferenceConfig": {
        "maxTokens": 512,
        "temperature": 0.7,
        "topP": 0.9
    }
}

model_id = "apac.amazon.nova-pro-v1:0"

response = bedrock.invoke_model(
    modelId=model_id,
    body=json.dumps(payload),
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response["body"].read())
response_text = response_body["output"]["message"]["content"][0]["text"]
print(response_text)