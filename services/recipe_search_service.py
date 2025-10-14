import numpy as np
from typing import List, Dict, Any
from services.recipe_service import normalize_ingredients, load_recipes_from_kb
from services.llm_parsing_service import llm_parsing_service
from utils import get_embedding
from models import RecipeRecommendation

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    try:
        a, b = np.array(vec_a), np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    except Exception as e:
        print(f"Error in cosine_similarity: {e}")
        return 0.0

def find_recipes_by_ingredients(user_ingredients: List[str], top_k: int = 5, threshold: float = 0.6) -> List[RecipeRecommendation]:
    """Find recipes based on available ingredients using semantic similarity."""
    try:
        # Load recipes from knowledge base
        recipes_data = load_recipes_from_kb()
        if not recipes_data:
            return []
        
        # Normalize user ingredients
        user_ingredients_normalized = []
        for ingredient in user_ingredients:
            normalized = normalize_ingredients(ingredient)
            user_ingredients_normalized.extend(normalized)
        
        # Check if we have embeddings available
        has_embeddings = any(
            recipe.get("embedding") is not None and 
            recipe.get("embedding") != [] and 
            len(recipe.get("embedding", [])) > 0
            for recipe in recipes_data
        )
        
        if not has_embeddings:
            print("⚠️  No embeddings available, falling back to keyword search")
            return find_recipes_by_keywords(user_ingredients, top_k)
        
        # Create query embedding from user ingredients
        query_text = ", ".join(user_ingredients_normalized)
        try:
            query_embedding = get_embedding(query_text)
        except Exception as e:
            print(f"⚠️  Failed to generate query embedding: {e}, falling back to keyword search")
            return find_recipes_by_keywords(user_ingredients, top_k)
        
        # Calculate similarity scores
        recommendations = []
        
        for recipe_data in recipes_data:
            if not recipe_data.get("embedding"):
                continue
            
            # Calculate semantic similarity
            similarity_score = cosine_similarity(query_embedding, recipe_data["embedding"])
            
            if similarity_score >= threshold:
                # Analyze ingredient matches
                recipe_ingredients = normalize_ingredients(recipe_data.get("ingredients", ""))
                
                # Find matched and missing ingredients
                matched_ingredients = []
                missing_ingredients = []
                
                for recipe_ingredient in recipe_ingredients:
                    is_matched = False
                    for user_ingredient in user_ingredients_normalized:
                        # Check for partial matches (substring or word overlap)
                        if (user_ingredient in recipe_ingredient or 
                            recipe_ingredient in user_ingredient or
                            any(word in recipe_ingredient.split() for word in user_ingredient.split())):
                            matched_ingredients.append(recipe_ingredient)
                            is_matched = True
                            break
                    
                    if not is_matched:
                        missing_ingredients.append(recipe_ingredient)
                
                # Calculate match ratio
                match_ratio = len(matched_ingredients) / len(recipe_ingredients) if recipe_ingredients else 0
                
                # Combine semantic similarity with ingredient match ratio
                combined_score = (similarity_score * 0.6) + (match_ratio * 0.4)
                
                recommendation = RecipeRecommendation(
                    recipe_id=recipe_data.get("recipe_id", ""),
                    title=recipe_data.get("title", ""),
                    ingredients=recipe_data.get("ingredients", ""),
                    steps=recipe_data.get("steps", ""),
                    match_score=combined_score,
                    matched_ingredients=matched_ingredients,
                    missing_ingredients=missing_ingredients
                )
                recommendations.append(recommendation)
        
        # Sort by combined score and return top results
        recommendations.sort(key=lambda x: x.match_score, reverse=True)
        return recommendations[:top_k]
        
    except Exception as e:
        print(f"Error in find_recipes_by_ingredients: {e}")
        return []

def find_recipes_by_keywords(user_ingredients: List[str], top_k: int = 5) -> List[RecipeRecommendation]:
    """Find recipes using keyword matching as fallback."""
    try:
        recipes_data = load_recipes_from_kb()
        if not recipes_data:
            return []
        
        # Normalize user ingredients
        user_ingredients_normalized = []
        for ingredient in user_ingredients:
            normalized = normalize_ingredients(ingredient)
            user_ingredients_normalized.extend(normalized)
        
        recommendations = []
        
        for recipe_data in recipes_data:
            recipe_ingredients = normalize_ingredients(recipe_data.get("ingredients", ""))
            
            # Count keyword matches
            matches = 0
            matched_ingredients = []
            missing_ingredients = []
            
            for recipe_ingredient in recipe_ingredients:
                is_matched = False
                for user_ingredient in user_ingredients_normalized:
                    if (user_ingredient in recipe_ingredient or 
                        recipe_ingredient in user_ingredient or
                        any(word in recipe_ingredient.split() for word in user_ingredient.split())):
                        matches += 1
                        matched_ingredients.append(recipe_ingredient)
                        is_matched = True
                        break
                
                if not is_matched:
                    missing_ingredients.append(recipe_ingredient)
            
            if matches > 0:
                match_score = matches / len(recipe_ingredients) if recipe_ingredients else 0
                
                recommendation = RecipeRecommendation(
                    recipe_id=recipe_data.get("recipe_id", ""),
                    title=recipe_data.get("title", ""),
                    ingredients=recipe_data.get("ingredients", ""),
                    steps=recipe_data.get("steps", ""),
                    match_score=match_score,
                    matched_ingredients=matched_ingredients,
                    missing_ingredients=missing_ingredients
                )
                recommendations.append(recommendation)
        
        # Sort by match score and return top results
        recommendations.sort(key=lambda x: x.match_score, reverse=True)
        return recommendations[:top_k]
        
    except Exception as e:
        print(f"Error in find_recipes_by_keywords: {e}")
        return []

def hybrid_recipe_search(user_ingredients: List[str], top_k: int = 5, threshold: float = 0.6) -> List[RecipeRecommendation]:
    """Combine semantic and keyword search for recipe recommendations."""
    try:
        # Try semantic search first
        semantic_results = find_recipes_by_ingredients(user_ingredients, top_k, threshold)
        
        # If we don't have enough results, supplement with keyword search
        if len(semantic_results) < top_k:
            keyword_results = find_recipes_by_keywords(user_ingredients, top_k)
            
            # Merge results, avoiding duplicates
            seen_ids = {r.recipe_id for r in semantic_results}
            for result in keyword_results:
                if result.recipe_id not in seen_ids:
                    semantic_results.append(result)
                    seen_ids.add(result.recipe_id)
                    if len(semantic_results) >= top_k:
                        break
        
        # Sort by match score and return top results
        semantic_results.sort(key=lambda x: x.match_score, reverse=True)
        return semantic_results[:top_k]
        
    except Exception as e:
        print(f"Error in hybrid_recipe_search: {e}")
        return []

def search_recipes_with_llm_parsing(user_input: str, top_k: int = 5, threshold: float = 0.6) -> List[RecipeRecommendation]:
    """
    Search for recipes using LLM parsing to extract ingredients from user input.
    
    Args:
        user_input: Raw user input describing ingredients and preferences
        top_k: Number of top results to return
        threshold: Similarity threshold for semantic search
        
    Returns:
        List of recipe recommendations
    """
    try:
        # Use LLM parsing service to extract ingredients from user input
        extracted_ingredients, preferences = llm_parsing_service.parse_user_input(user_input)
        
        print(f"LLM Parsed ingredients: {extracted_ingredients}")
        print(f"LLM Parsed preferences: {preferences}")
        
        # If no ingredients were extracted, try fallback parsing
        if not extracted_ingredients:
            print("No ingredients extracted by LLM, falling back to input parsing")
            # Split the input by common delimiters as fallback
            extracted_ingredients = [ing.strip() for ing in user_input.split(',') if ing.strip()]
        
        # Use the extracted ingredients for hybrid search
        if extracted_ingredients:
            recommendations = hybrid_recipe_search(extracted_ingredients, top_k, threshold)
            
            # Add preferences context to recommendations if available
            if preferences:
                print(f"User preferences noted: {preferences}")
                # Preferences could be used for future filtering or ranking
            
            return recommendations
        else:
            print("No ingredients found in user input")
            return []
            
    except Exception as e:
        print(f"Error in search_recipes_with_llm_parsing: {e}")
        # Fallback to direct input parsing
        fallback_ingredients = [ing.strip() for ing in user_input.split(',') if ing.strip()]
        if fallback_ingredients:
            return hybrid_recipe_search(fallback_ingredients, top_k, threshold)
        return []
