import clip
import torch
from PIL import Image
import pandas as pd

class Quirky:
    def __init__(self,csv_file_path,caption_column):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.csv_file_path = csv_file_path
        self.caption_column = caption_column
    
    #Encode text to CLIP vector
    def encode_text(self,search_query):
        with torch.no_grad():
            text_encoded = self.model.encode_text(clip.tokenize(search_query).to(self.device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        return text_encoded
    
    def encode_image(self,image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
        
    def get_similarity(self,search_query, image_path):
        photo_features = self.encode_image(image_path)
        text_features = self.encode_text(search_query)
        return (photo_features @ text_features.T).squeeze(1).item()
    
    def generate_vector_database(self):
        df = pd.read_csv(self.csv_file_path)
        df['Vec'] = df[self.caption_column].apply(lambda x: self.encode_text(x))
        tensor_list = [tensor for tensor in df['Vec']]
        vector_db = torch.stack(tensor_list)
        torch.save(vector_db, 'vector_db_1.pt')

    def get_dataframe_with_vectors(self):
        df = pd.read_csv(self.csv_file_path)
        vector_db = torch.load('vector_db_1.pt')
        df['Vec'] = [i for i in vector_db]
        return df
    
    def get_single_vector(self,image_path,text = ''):
        features = self.encode_image(image_path)
        if(text != ''):
            features += self.encode_text(text)
        return features
    
    def get_most_similar_caption_from_text_image(self,image_path,df,text = ''):
        features = self.get_single_vector(image_path,text)
        df = self.get_dataframe_with_vectors()
        df['Similarity'] = df['Vec'].apply(lambda x: (features @ x.T).squeeze(1).item())
        return df.sort_values(by=['Similarity'],ascending=False)


