import customtkinter,openai,autocomplete # Importing the necessary libraries
import torch 
from transformers import AutoTokenizer, AutoModelWithLMHead,pipeline

def local_summarize():
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
    data= textbox.get('1.0','end')
    " ".join(data)
    inputs = tokenizer.encode("Summarize:" + data, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(inputs,max_length=150,min_length=80, length_penalty=5.,num_beams=2)
    summary=tokenizer.decode(outputs[0])
    summary = summary[5:-4]
    summary_var.set(summary)

def local_summarize2():
    data= textbox.get('1.0','end')
    " ".join(data)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary_list=summarizer(data, max_length=130, min_length=30, do_sample=False)
    summary_dictionary=summary_list[0]
    summary=summary_dictionary['summary_text']
    summary_var.set(summary)

def chat_gpt_summarizer(): 
    API_KEY=open("Api_key","r").read()
    api_key=API_KEY
    text_data= textbox.get('1.0','end')

    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Summarize the following text:\n{text_data}\n\nSummary:",
        max_tokens=50,  # Adjust the number of tokens as needed for the desired summary length
        api_key=api_key
    )
    # Extract and print the generated summary
    global summary
    summary = response.choices[0].text.strip()
    print(summary)
    summary_var.set(summary)


app = customtkinter.CTk()
app.title("Wordy")
app.geometry("1920x1080")
def get_text():
    text_data= textbox.get('1.0','end')
    with open("output.txt","w+") as f:
        f.write(text_data)
        f.close()
def explorer():
    file=customtkinter.filedialog.askopenfile(filetypes=[("Text Files", "*.txt")])
    try:
        textbox.delete(0.0,'end')
        textbox.insert(0.0,file.read())
    except:
        print("File Error")
def autocompleter(event=None):
    text = textbox.get("0.0", "end")  
    words = text.split()
    if len(words)>=2:
        current_word = words[-1]
        previous_word = words[-2]
        suggestions = autocomplete.predict(previous_word,current_word)
        topsuggestion=suggestions[0]
        if suggestions:
            try:
                global topsuggestion_refined
                if len(topsuggestion)==0:
                    print("no suggestion")
                    autocomplete_var.set(" ")
                else:
                    autocomplete_var.set(topsuggestion[0])
                    topsuggestion_refined=topsuggestion[0]
            except:
                print("no suggestion")
                autocomplete_var.set(" ")

def autocomplete_initializer():
    f=open("Output_converted.txt","r",encoding="utf8")
    data=f.read()
    import autocomplete 
    from autocomplete import models
    models.train_models(data)
    autocomplete.load()



def accept_suggestion(event=None):
    print("suggestion accepted")
    current_text = textbox.get("1.0", "end-1c")  # Get the current text content
    lines = current_text.splitlines()
    if len(lines) > 0:
    # Split the last line into words using spaces as separators
        last_line_words = lines[-1].split()

    # Check if there are words in the last line
        if len(last_line_words) > 0:
            # Pop the last word from the last line
            last_line_words.pop()

            # Append a new word
            new_word = topsuggestion_refined

            # Add the new word to the last line
            last_line_words.append(new_word)

            # Join the words back together into a modified last line
            modified_last_line = " ".join(last_line_words)

            # Replace the last line with the modified last line in the list of lines
            lines[-1] = modified_last_line

            # Join the lines back together with newline characters while preserving formatting
            modified_text = "\n".join(lines)

            textbox.delete("1.0", "end")  # Delete the current text
            textbox.insert("1.0", modified_text)  # Insert the modified tet
            




   ## words = current_text.split()  # Split the text into words #Legacy code #
   ## if not words:
   ##     return
  ##  words[len(words)-1] = topsuggestion_refined  # Replacing with word
   ## modified_text=" ".join(words)
    ##textbox.delete("1.0", "end")  # Delete the current text
    ##textbox.insert("1.0", modified_text)  # Insert the modified tet




autocomplete_initializer()



autocomplete_var= customtkinter.StringVar()
autocomplete_label = customtkinter.CTkLabel(app, text="Autocomplete:")
autocomplete_label.pack()
autocomplete_display = customtkinter.CTkLabel(app, textvariable=autocomplete_var)
autocomplete_display.pack() 

summary_var = customtkinter.StringVar()
summary_label = customtkinter.CTkLabel(app, text="Summary:")
summary_label.pack()
summary_display = customtkinter.CTkLabel(app, textvariable=summary_var,wraplength=300,justify='center')
summary_display.pack()


textbox = customtkinter.CTkTextbox(app,height=480,width=1280)
textbox.bind("<KeyRelease>",autocompleter)
textbox.bind("<2>",accept_suggestion)
textbox.pack(pady=30, padx=20)

button = customtkinter.CTkButton(app,text="Save", command=get_text)
button.pack(pady=10, padx=20)

button_explore = customtkinter.CTkButton(app,text="Open", command=explorer)
button_explore.pack(pady=10,padx=20)

summarize_button = customtkinter.CTkButton(app, text="GPT 3.5 Summarize", command=chat_gpt_summarizer)
summarize_button.pack(pady=10,padx=20)

summarize_button2 = customtkinter.CTkButton(app, text="Local Summarize Google T5", command=local_summarize)
summarize_button2.pack(pady=10,padx=20)

summarize_button3 = customtkinter.CTkButton(app, text="BART Summarizer", command=local_summarize2)
summarize_button3.pack(pady=10,padx=20)

app.mainloop()

