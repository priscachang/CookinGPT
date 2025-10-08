import csv
import json
import os
from typing import List, Dict, Any
from models import Recipe
from utils import get_embedding

def process_csv_recipes(csv_file_path: str) -> List[Recipe]:
    """Process CSV file and extract recipe data."""
    recipes = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            # Try to detect delimiter
            sample = file.read(1024)
            file.seek(0)
            
            # Common delimiters to try
            delimiters = [',', ';', '\t', '|']
            detected_delimiter = ','
            
            for delimiter in delimiters:
                if delimiter in sample:
                    detected_delimiter = delimiter
                    break
            
            reader = csv.DictReader(file, delimiter=detected_delimiter)
            
            for row_num, row in enumerate(reader, 1):
                try:
                    # Extract required fields (case-insensitive)
                    recipe_id = str(row.get('id', f'recipe_{row_num}')).strip()
                    title = str(row.get('title', '')).strip()
                    ingredients = str(row.get('ingredient', row.get('ingredients', ''))).strip()
                    steps = str(row.get('step', row.get('steps', ''))).strip()
                    
                    if not title or not ingredients:
                        print(f"Skipping row {row_num}: Missing required fields (title or ingredients)")
                        continue
                    
                    recipe = Recipe(
                        id=recipe_id,
                        title=title,
                        ingredients=ingredients,
                        steps=steps
                    )
                    recipes.append(recipe)
                    
                except Exception as e:
                    print(f"Error processing row {row_num}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise
    
    return recipes

def generate_ingredient_embeddings(recipes: List[Recipe]) -> List[Recipe]:
    """Generate embeddings for recipe ingredients."""
    recipes_with_embeddings = []
    
    for recipe in recipes:
        try:
            # Create embedding from ingredients text
            embedding = get_embedding(recipe.ingredients)
            recipe.embedding = embedding
            recipes_with_embeddings.append(recipe)
        except Exception as e:
            print(f"Error generating embedding for recipe {recipe.id}: {e}")
            # Still add recipe without embedding
            recipes_with_embeddings.append(recipe)
    
    return recipes_with_embeddings

def save_recipes_to_kb(recipes: List[Recipe], kb_file: str = "recipe_knowledge_base.json"):
    """Save recipes to knowledge base file."""
    try:
        # Convert recipes to dictionary format for JSON serialization
        recipes_data = []
        for recipe in recipes:
            recipe_dict = {
                "recipe_id": recipe.id,
                "title": recipe.title,
                "ingredients": recipe.ingredients,
                "steps": recipe.steps,
                "embedding": recipe.embedding,
                "metadata": {
                    "type": "recipe",
                    "ingredient_count": len(recipe.ingredients.split(',')) if recipe.ingredients else 0
                }
            }
            recipes_data.append(recipe_dict)
        
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(recipes_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(recipes_data)} recipes to {kb_file}")
        return True
        
    except Exception as e:
        print(f"Error saving recipes to knowledge base: {e}")
        return False

def load_recipes_from_kb(kb_file: str = "recipe_knowledge_base.json") -> List[Dict[str, Any]]:
    """Load recipes from knowledge base file."""
    try:
        if not os.path.exists(kb_file):
            return []
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            recipes_data = json.load(f)
        
        return recipes_data
        
    except Exception as e:
        print(f"Error loading recipes from knowledge base: {e}")
        return []

def normalize_ingredients(ingredients_text: str) -> List[str]:
    """Normalize and clean ingredient list."""
    if not ingredients_text:
        return []
    
    # Split by common separators
    ingredients = []
    for separator in [',', ';', '\n', '|']:
        if separator in ingredients_text:
            ingredients = [ing.strip().lower() for ing in ingredients_text.split(separator)]
            break
    
    if not ingredients:
        ingredients = [ingredients_text.strip().lower()]
    
    # Clean up ingredients
    cleaned_ingredients = []
    for ingredient in ingredients:
        if ingredient:
            # Remove common prefixes/suffixes
            ingredient = ingredient.strip()
            # Remove quantities (basic pattern)
            import re
            ingredient = re.sub(r'^\d+\s*', '', ingredient)  # Remove leading numbers
            ingredient = re.sub(r'\s+\d+\s*$', '', ingredient)  # Remove trailing numbers
            ingredient = ingredient.strip()
            if ingredient:
                cleaned_ingredients.append(ingredient)
    
    return cleaned_ingredients
