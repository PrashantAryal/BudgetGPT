from flask import Flask, request, render_template
from ai_model import qa_chain_en, qa_chain_np
from langdetect import detect
app = Flask(__name__)

# Storing chat history and user information
chat_history = []
user_info = {}
chat_state = 'idle' 


@app.route('/', methods=['GET', 'POST'])
def handle_query():
    global chat_history
    if request.method == 'POST':
        query = request.form['query'].strip()

        # Handle greetings
        greetings = {"Hi", "Hello", "Good morning", "Good evening"}
        if query in greetings:
            response = "Hi, How can I assist you?"
            chat_history.append({'query': query, 'response': response, 'sources': []})
            return render_template('index.html', chat_history=chat_history)

        detected_lang = detect(query)
        print(detected_lang)

        if detected_lang == 'ne':
            result = qa_chain_np({"query":query})
        elif detected_lang =='en':
            result = qa_chain_en({"query":query})

        response = result['result']
        
        chat_history.append({
            'query': query,
            'response': response,
        })
        
        return render_template('index.html', chat_history=chat_history)
    
    return render_template('index.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)