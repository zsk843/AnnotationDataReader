from DataReader import TextData

data = TextData()
data.load_from_raw(r"D:\studentGroup\training_data")
fun_lst = [TextData.remove_punctuation, TextData.remove_stop_words, TextData.replace_hidden_words, TextData.original_form]
label_fun_lst = [TextData.rule_employment, TextData.rule_mobility]
data.label_processing(label_fun_lst)
data.text_processing(fun_lst)
data.save()

