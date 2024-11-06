from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import pymysql
import pandas as pd
from sqlalchemy import create_engine


class ImageDescriptionGenerator:
    # Load model and processor
    def __init__(self, model_id="openai/clip-vit-base-patch16"):
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    # Load image and generate inputs
    def input_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs

    # Model inference to get feature vector
    def image_get_features(self, image_path):
        inputs = self.input_image(image_path)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize image embedding vector
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    # Batch process images to generate feature vectors and update the database
    def batch_image_get_features(self):
        db_host = 'localhost'  # Database host name
        db_user = 'tester_1'  # MySQL username
        db_password = '123'  # MySQL password
        db_name = 'tester_image'  # Database name
        table_name = 'image'

        # Create SQLAlchemy engine with specified SQL dialect
        connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
        engine = create_engine(connection_string, echo=True)

        with engine.connect() as connection:
            read_query = f"SELECT id, path FROM {table_name}"  # Replace with your query
            df = pd.read_sql(read_query, connection)

        if df.empty:
            print("No images to process")
            return

        # Create a list to store update data
        updates = []

        # Process each image
        for _, row in df.iterrows():
            image_id = row['id']
            path = row['path']
            try:
                # Get image features
                image_feature = self.image_get_features(image_path=path)
                # Convert tensor to list and then to string for storage
                feature_str = ",".join(map(str, image_feature.squeeze().tolist()))
                updates.append((feature_str, image_id))
            except Exception as e:
                print(f"Error processing image {path}: {e}")

        # If there is data to update, execute the update
        if updates:
            update_query = "UPDATE image SET description = %s WHERE id = %s"

            # Use pymysql connection to execute batch update
            connection = pymysql.connect(
                host=db_host,
                user=db_user,
                password=db_password,
                database=db_name
            )
            try:
                with connection.cursor() as cursor:
                    cursor.executemany(update_query, updates)
                connection.commit()
                print(f"Successfully updated {len(updates)} records")
            except Exception as e:
                print(f"Error updating the database: {e}")
            finally:
                connection.close()


if __name__ == "__main__":
    image_description_generator = ImageDescriptionGenerator()
    image_description_generator.batch_image_get_features()
