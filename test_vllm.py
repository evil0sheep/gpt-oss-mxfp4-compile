from openai import OpenAI
import sys

def main():
    print("Initializing OpenAI client pointing to vLLM...")
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )

    model_name = "openai/gpt-oss-20b"
    print(f"Sending request to model: {model_name}")

    try:
        result = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you tell me what model you are?"}
            ]
        )

        print("\nResponse received:")
        print("-" * 50)
        print(result.choices[0].message.content)
        print("-" * 50)

    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
