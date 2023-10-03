import customtkinter,openai,autocomplete,torch # Importing the necessary libraries
from transformers import AutoTokenizer, AutoModelWithLMHead,pipeline
display_width=1920
display_height=1080
text_box_height=720
text_box_width=1280
save_file_name="Wordy_Document"
class App(customtkinter.CTk):
    display_width=1920
    display_height=1080
    text_box_height=720
    text_box_width=1280
    


    
    
    def file_menu_action(self,value):
        if value=="Open":
            print("Opening")
            self.file_open()
            #Run open script
        elif value=="Save":
            self.saver()
            #Run save script
        else:
            print("File Drop Down Menu Error")
    
    def summary_menu_action(self,value):
        if value=="GPT 3.5 Summarize":
            print("GPT SUMMARY")
            self.chat_gpt_summarizer()
        elif value=="BART Summarize":
            print("BART SUMMARY")
            self.BART_summarize()
        elif value=="Google T5 Summarize":
            print("T5 SUMMARY")
            self.t5_summarize()
   
    def file_open(self):
     file=customtkinter.filedialog.askopenfile(filetypes=[("Text Files", "*.txt")])
     try:
         self.textbox.delete(0.0,'end')
         self.textbox.insert(0.0,file.read())
     except:
         print("File Error")
    
    def saver(self,event=None):
        text_data= self.textbox.get('1.0','end')
        save_file_name= self.file_name_textbox.get()
        save_file_name=str(save_file_name+".txt")
        with open(save_file_name,"w") as f:
            f.write(text_data)
            f.close()
        print("Saved")
    def autocompleter(self,event=None):
        text = self.textbox.get("0.0", "end")  
        words = text.split()
        try:
            if len(words)>=2:
                current_word = words[-1]
                previous_word = words[-2]
                suggestions = autocomplete.predict(previous_word,current_word)
                if len(suggestions)!=0:
                    topsuggestion=suggestions[0]
                    if suggestions:
                        try:
                            if len(topsuggestion)==0:
                                print("no suggestion")
                                self.autocomplete_var.set("Autocomplete: ")
                            elif len(topsuggestion)!=0:
                                global topsuggestion_refined
                                self.autocomplete_var.set(f"Autocomplete: {topsuggestion[0]}")
                                topsuggestion_refined=topsuggestion[0]
                                return topsuggestion_refined
                            else:
                                print("no suggestion")
                                self.autocomplete_var.set("Autocomplete: ")
                        except:
                            print("no suggestion")
                            self.autocomplete_var.set("Autocomplete: ")
                else:
                    self.autocomplete_var.set("Autocomplete: ")
            else:
                self.autocomplete_var.set("Autocomplete: ")
        except:
            print("Unsupported character/key, skipping autocomplete generation")


    def autocomplete_initializer():
        f=open("Output_converted.txt","r",encoding="utf8")
        data=f.read()
        from autocomplete import models
        models.train_models(data)
    autocomplete.load()



    def accept_suggestion(self,event=None):
        current_text = self.textbox.get("1.0", "end-1c")  # Get the current text content
        lines = current_text.splitlines()
        if len(lines) > 0:
        # Split the last line into words using spaces as separators
            last_line_words = lines[-1].split()

        # Check if there are words in the last line
            if len(last_line_words) > 0 and topsuggestion_refined:
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

                self.textbox.delete("1.0", "end")  # Delete the current text
                self.textbox.insert("1.0", modified_text)  # Insert the modified tet
                print(f"Suggestion: '{topsuggestion_refined}' accepted")

    def autocomplete_initializer():
        f=open("Output_converted.txt","r",encoding="utf8")
        data=f.read()
        import autocomplete 
        from autocomplete import models
        models.train_models(data)
        autocomplete.load()
    
    autocomplete_initializer()
    
    def t5_summarize(self):
        tokenizer = AutoTokenizer.from_pretrained('t5-base')
        model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
        data= self.textbox.get('1.0','end')
        " ".join(data)
        inputs = tokenizer.encode("Summarize:" + data, return_tensors='pt', max_length=512, truncation=True)
        outputs = model.generate(inputs,max_length=150,min_length=80, length_penalty=5.,num_beams=2)
        summary=tokenizer.decode(outputs[0])
        summary = summary[5:-4]
        self.summary_var.set(f"Summary: {summary}")   

    def BART_summarize(self):
        data= self.textbox.get('1.0','end')
        " ".join(data)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary_list=summarizer(data, max_length=130, min_length=30, do_sample=False)
        summary_dictionary=summary_list[0]
        summary=summary_dictionary['summary_text']
        self.summary_var.set(f"Summary: {summary}")

    def chat_gpt_summarizer(self): 
        try:
            API_KEY=open("Api_key","r").read()
            api_key=API_KEY
            text_data= self.textbox.get('1.0','end')

            if len(api_key)!=0:
                try:
                    response = openai.Completion.create(
                        engine="davinci",
                        prompt=f"Summarize the following text:\n{text_data}\n\nSummary:",
                        max_tokens=50,  # Adjust the number of tokens as needed for the desired summary length
                        api_key=api_key
                    )
                    # Extract and print the generated summary
                    summary = response.choices[0].text.strip()
                    print(summary)
                    self.summary_var.set(f"Summary: {summary}")
                except:
                    print("GPT error, check credit availablity or API key")
            else:
                print("Api Key not loaded")
        except:
            print("API key file read error")
   
    def __init__(self):
        super().__init__()
        self.geometry(f"{display_width}*{display_height}")
        self.grid_columnconfigure(5, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.title("Wordy!")
       
    
        self.file_menu=customtkinter.CTkOptionMenu(self,values=["Save","Open"],command=self.file_menu_action)
        self.file_menu.set("Save")
        self.file_menu.grid(row=0,column=0,padx=20,pady=20,columnspan=1,sticky='w')

        self.summary_menu=customtkinter.CTkOptionMenu(self,values=["GPT 3.5 Summarize","BART Summarize","Google T5 Summarize"],command=self.summary_menu_action)
        self.summary_menu.set("BART Summarize")
        self.summary_menu.grid(row=0,column=1,padx=20,pady=20,columnspan=2,sticky='w')

        self.textbox=customtkinter.CTkTextbox(self,height=text_box_height,width=text_box_width,wrap='word')
        self.textbox.bind("<KeyRelease>",self.autocompleter)
        self.textbox.bind("<2>",self.accept_suggestion)
        self.textbox.grid(row=3,column=0,padx=20,pady=0,sticky="ew",columnspan=5)

        self.file_name_textbox=customtkinter.CTkEntry(self,height=14,placeholder_text="Enter File Name To Save As",corner_radius=10)
        self.file_name_textbox.grid(row=0,column=3,padx=20,pady=20,sticky="ew",columnspan=2)
        self.file_name_textbox.bind("<Return>",self.saver)
        

        self.summary_var=customtkinter.StringVar()
        self.summary_display = customtkinter.CTkLabel(self, textvariable=self.summary_var,wraplength=800,justify='left')
        self.summary_var.set("Summary: ")
        self.summary_display.grid(row=1,column=0,padx=20,pady=20,columnspan=5,sticky='w')
        
        self.autocomplete_var=customtkinter.StringVar()
        self.autocomplete_display = customtkinter.CTkLabel(self, textvariable=self.autocomplete_var,wraplength=display_width)
        self.autocomplete_var.set("Autocomplete: ")
        self.autocomplete_display.grid(row=2,column=0,padx=20,pady=20,columnspan=1,sticky='w')
app= App()
app.mainloop()
