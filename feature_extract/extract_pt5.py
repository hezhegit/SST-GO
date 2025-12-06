import glob
import os
import re
import torch
import pickle
from transformers import T5Tokenizer, T5EncoderModel
from Bio import SeqIO


torch.hub.set_dir('./th_hub')

ont = 'bp'

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 指定模型保存目录
model_dir = "./th_hub"
os.makedirs(model_dir, exist_ok=True)

# 下载并保存分词器和模型到指定目录
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir=model_dir, do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir=model_dir).to(device)
model = model.eval()

# 读取FASTA文件中的序列信息
fasta_file = "/home/415/hz_project/HEAL/data/nrPDB-GO_2019.06.18_val_sequences.fasta"
sequences = []

# 已经生成的id
# pt_id = []
# on_save = '/home/415/hz_project/MMSMAPlus-master/data/pt5/cafa3'
# pt_files = glob.glob(os.path.join(on_save, '*.pkl'))
# for file_path in pt_files:
#     sample_id = os.path.basename(file_path).split('.')[0]
#     pt_id.append(sample_id)



for record in SeqIO.parse(fasta_file, "fasta"):
    sequences.append((record.id, str(record.seq)))

# 处理序列并提取特征
save_dir = "/home/415/hz_project/data_processed/pt5/pdb/"
os.makedirs(save_dir, exist_ok=True)

pt5_emb = dict()

for label, sequence in sequences:
    # 已经生成的，跳过
    # if str(label) in pt_id:
    #     continue

    if len(sequence) > 2000:
        sequence = sequence[:2000]

    # 替换稀有/模糊的氨基酸为X，并在所有氨基酸之间添加空格
    sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))

    # 对序列进行分词并进行填充
    ids = tokenizer(sequence, add_special_tokens=True, padding="longest", return_tensors="pt")
    input_ids = ids['input_ids'].to(device)
    attention_mask = ids['attention_mask'].to(device)

    try:
        # 生成嵌入
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        # 获取嵌入表示
        embeddings = embedding_repr.last_hidden_state.squeeze(0).cpu().numpy()

        # 保存嵌入到指定目录
        pt5_emb[label] = embeddings[:-1,:]
        # with open(os.path.join(save_dir, f'{label}.pkl'), 'wb') as f:
        #     pickle.dump({"pt5": embeddings}, f)

        print(f"{label} successful!")

    except torch.cuda.OutOfMemoryError:
        print(f"{label} failed due to memory error. Skipping this sequence.")

    # 清理缓存
    torch.cuda.empty_cache()

with open(save_dir + "pdb_val.pkl", "wb") as f:
    pickle.dump(pt5_emb, f)

print("Feature extraction and saving complete.")
