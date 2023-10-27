import json

data_path = 'log_data/ad_data/log_ad_alpaca_w-10.json'
with open(data_path, 'r') as file:
        data = json.load(file)

print(len(data))
print(data[0])
input_len = 0
output_len = 0
for line in data:
    input_len += len(line['conversation'][0]['input'].split())
    output_len += len(line['conversation'][0]['output'].split())
avg_input_len = input_len / len(data)
avg_output_len = output_len / len(data)
print(avg_input_len)
print(avg_output_len)
    