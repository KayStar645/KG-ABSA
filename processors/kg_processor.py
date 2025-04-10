import networkx as nx
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional
from sklearn_crfsuite import CRF
import numpy as np
from sklearn.model_selection import train_test_split

class AspectExtractor:
    """
    Lớp trích xuất term và gán khía cạnh sử dụng PhoBERT + CRF
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = AutoModel.from_pretrained("vinai/phobert-base")
        self.crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
        self.is_trained = False
        
    def train(self, texts: List[str], labels: List[List[str]]):
        """
        Huấn luyện CRF với dữ liệu đã gán nhãn
        Args:
            texts (List[str]): Danh sách các văn bản
            labels (List[List[str]]): Danh sách các nhãn tương ứng
        """
        # Chia dữ liệu thành tập train và validation
        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
        
        # Trích xuất features cho tập train
        train_features = []
        for text in X_train:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            features = self._get_features(embeddings)
            train_features.append(features)
        
        # Huấn luyện CRF
        self.crf.fit(train_features, y_train)
        self.is_trained = True
        
    def extract_terms(self, text: str) -> List[str]:
        """
        Trích xuất các term từ văn bản sử dụng PhoBERT
        Args:
            text (str): Văn bản đầu vào
        Returns:
            List[str]: Danh sách các term được trích xuất
        """
        if not self.is_trained:
            raise ValueError("CRF chưa được huấn luyện. Vui lòng gọi phương thức train() trước.")
            
        # Tokenize và encode văn bản
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Lấy embeddings từ lớp cuối cùng
        embeddings = outputs.last_hidden_state
        
        # Sử dụng CRF để dự đoán nhãn cho mỗi token
        features = self._get_features(embeddings)
        labels = self.crf.predict([features])[0]
        
        # Trích xuất các term dựa trên nhãn
        terms = []
        current_term = []
        for token, label in zip(self.tokenizer.tokenize(text), labels):
            if label == 'B-TERM':
                if current_term:
                    terms.append(''.join(current_term))
                current_term = [token.replace('▁', '')]
            elif label == 'I-TERM':
                current_term.append(token.replace('▁', ''))
            else:
                if current_term:
                    terms.append(''.join(current_term))
                current_term = []
        if current_term:
            terms.append(''.join(current_term))
            
        return terms
    
    def _get_features(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Trích xuất features từ embeddings cho CRF
        Args:
            embeddings (torch.Tensor): Embeddings từ PhoBERT
        Returns:
            np.ndarray: Features cho CRF
        """
        # Lấy features từ embeddings (có thể thêm các features khác)
        return embeddings.mean(dim=1).numpy()

class KnowledgeGraph:
    """
    Lớp quản lý Knowledge Graph cho phân tích cảm xúc
    """
    
    def __init__(self, aspect_columns: List[str]):
        """
        Khởi tạo Knowledge Graph từ dữ liệu
        Args:
            aspect_columns (List[str]): Danh sách các cột aspect từ CSV
        """
        self.G = nx.DiGraph()
        self.aspect_hierarchy = {}
        
        # Xây dựng cấu trúc phân cấp từ tên các aspect
        for aspect in aspect_columns:
            if '#' in aspect:
                parent, child = aspect.split('#')
                self.aspect_hierarchy[aspect] = parent
                self.G.add_edge(child, parent, relation="belongs-to")
    
    def get_parent_aspect(self, aspect_category: str) -> Optional[str]:
        """
        Lấy aspect cha của một aspect category
        Args:
            aspect_category (str): Aspect category cần tìm cha
        Returns:
            Optional[str]: Aspect cha nếu tìm thấy, None nếu không
        """
        return self.aspect_hierarchy.get(aspect_category)
        
    def infer_related_aspects(self, aspect_category: str, sentiment: str) -> List[Tuple[str, str]]:
        """
        Suy luận các aspect liên quan dựa trên aspect và sentiment hiện tại
        Args:
            aspect_category (str): Aspect category hiện tại
            sentiment (str): Sentiment của aspect
        Returns:
            List[Tuple[str, str]]: Danh sách các aspect liên quan và sentiment dự đoán
        """
        related_aspects = []
        parent_aspect = self.get_parent_aspect(aspect_category)
        
        if parent_aspect:
            # Tìm tất cả các aspect con của cùng aspect cha
            for aspect in self.aspect_hierarchy:
                if self.aspect_hierarchy[aspect] == parent_aspect and aspect != aspect_category:
                    # Nếu sentiment tiêu cực, có thể ảnh hưởng đến các aspect khác
                    if sentiment == 'negative':
                        related_aspects.append((aspect, 'negative'))
                    # Nếu sentiment tích cực, có thể ảnh hưởng đến các aspect liên quan
                    elif sentiment == 'positive':
                        related_aspects.append((aspect, 'positive'))
                    
        return related_aspects

def process_with_kg(csv_file: str, output_file: str):
    """
    Xử lý dữ liệu CSV với PhoBERT + CRF và Knowledge Graph
    Args:
        csv_file (str): Đường dẫn đến file CSV đầu vào
        output_file (str): Đường dẫn đến file CSV đầu ra
    """
    # Khởi tạo bộ trích xuất term
    extractor = AspectExtractor()
    
    # Đọc dữ liệu
    df = pd.read_csv(csv_file)
    
    # Lấy danh sách các cột aspect
    aspect_columns = [col for col in df.columns if col != 'Review']
    
    # Khởi tạo Knowledge Graph từ dữ liệu
    kg = KnowledgeGraph(aspect_columns)
    
    # Chuẩn bị dữ liệu huấn luyện
    texts = []
    labels = []
    for _, row in df.iterrows():
        review = row['Review']
        texts.append(review)
        # Tạo nhãn cho mỗi token (O: không phải term, B-TERM: bắt đầu term, I-TERM: tiếp tục term)
        review_labels = ['O'] * len(review.split())
        for i, word in enumerate(review.split()):
            for aspect in aspect_columns:
                if word.lower() in aspect.lower():
                    review_labels[i] = 'B-TERM' if i == 0 or review_labels[i-1] == 'O' else 'I-TERM'
        labels.append(review_labels)
    
    # Huấn luyện CRF
    extractor.train(texts, labels)
    
    # Tạo cột mới cho các khía cạnh được dự đoán
    df['predicted_aspects'] = None
    
    # Xử lý từng dòng
    for idx, row in df.iterrows():
        review = row['Review']
        predicted_aspects = []
        
        # Trích xuất các term từ review
        terms = extractor.extract_terms(review)
        
        # Tìm các khía cạnh hiện có và sentiment của chúng
        existing_aspects = {}
        for col in aspect_columns:
            sentiment_value = row[col]
            if sentiment_value != 0:  # Nếu có khía cạnh này
                sentiment = 'positive' if sentiment_value == 1 else 'negative' if sentiment_value == 2 else 'neutral'
                existing_aspects[col] = sentiment
        
        # Duyệt qua từng term đã trích xuất
        for term in terms:
            # Tìm aspect tương ứng với term
            for aspect in aspect_columns:
                if term.lower() in aspect.lower():
                    # Nếu aspect này chưa có trong dữ liệu
                    if aspect not in existing_aspects:
                        # Tìm các aspect liên quan đã có
                        related_aspects = []
                        for existing_aspect, existing_sentiment in existing_aspects.items():
                            # Kiểm tra xem aspect hiện tại có liên quan đến aspect đang xét không
                            if kg.get_parent_aspect(existing_aspect) == kg.get_parent_aspect(aspect):
                                related_aspects.append((existing_aspect, existing_sentiment))
                        
                        # Nếu có aspect liên quan, dự đoán sentiment cho aspect mới
                        if related_aspects:
                            # Lấy sentiment phổ biến nhất từ các aspect liên quan
                            sentiments = [s for _, s in related_aspects]
                            predicted_sentiment = max(set(sentiments), key=sentiments.count)
                            predicted_aspects.append((aspect, predicted_sentiment))
        
        # Lưu kết quả dự đoán
        if predicted_aspects:
            df.at[idx, 'predicted_aspects'] = ', '.join([f"{a}:{s}" for a, s in predicted_aspects])
    
    # Lưu kết quả
    df.to_csv(output_file, index=False)
    print(f"Đã xử lý và lưu kết quả vào: {output_file}")

if __name__ == '__main__':
    # Ví dụ sử dụng
    input_file = 'datasets/mmlab_uit_hotel/2-mmlab-uit-hotel-dev.csv'
    output_file = 'datasets_kg/mmlab_uit_hotel/2-mmlab-uit-hotel-dev.csv'
    process_with_kg(input_file, output_file) 