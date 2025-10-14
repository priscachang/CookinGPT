"""
Embedded recipe database for the Smart Recipe Finder application.
This module contains a curated collection of recipes that are pre-loaded
into the application, eliminating the need for users to upload CSV files.
"""

from typing import List, Dict, Any
from models import Recipe
from utils import get_embedding

# Embedded recipe data
EMBEDDED_RECIPES_DATA = [
    {
        "id": "recipe_001",
        "title": "Classic Spaghetti Carbonara",
        "ingredients": "spaghetti, eggs, parmesan cheese, pancetta, black pepper, salt, olive oil",
        "steps": "1. Cook spaghetti according to package directions. 2. In a bowl, whisk eggs with parmesan and black pepper. 3. Cook pancetta in olive oil until crispy. 4. Toss hot pasta with pancetta, then with egg mixture. 5. Serve immediately with extra parmesan."
    },
    {
        "id": "recipe_002", 
        "title": "Chicken Stir Fry",
        "ingredients": "chicken breast, bell peppers, broccoli, soy sauce, garlic, ginger, vegetable oil, rice",
        "steps": "1. Cut chicken into strips and marinate in soy sauce. 2. Heat oil in wok and cook chicken until done. 3. Add garlic and ginger, then vegetables. 4. Stir fry until vegetables are tender-crisp. 5. Serve over rice."
    },
    {
        "id": "recipe_003",
        "title": "Vegetarian Pasta Primavera",
        "ingredients": "pasta, zucchini, bell peppers, cherry tomatoes, garlic, olive oil, parmesan cheese, basil",
        "steps": "1. Cook pasta according to package directions. 2. Sauté garlic in olive oil. 3. Add vegetables and cook until tender. 4. Toss with pasta and parmesan. 5. Garnish with fresh basil."
    },
    {
        "id": "recipe_004",
        "title": "Beef Tacos",
        "ingredients": "ground beef, taco shells, lettuce, tomatoes, cheese, sour cream, taco seasoning, onions",
        "steps": "1. Brown ground beef with onions. 2. Add taco seasoning and water, simmer. 3. Warm taco shells. 4. Fill shells with beef mixture. 5. Top with lettuce, tomatoes, cheese, and sour cream."
    },
    {
        "id": "recipe_005",
        "title": "Salmon with Roasted Vegetables",
        "ingredients": "salmon fillets, sweet potatoes, broccoli, olive oil, lemon, herbs, salt, pepper",
        "steps": "1. Preheat oven to 400°F. 2. Toss vegetables with olive oil and seasonings. 3. Roast vegetables for 20 minutes. 4. Add salmon to pan and roast 12-15 minutes. 5. Serve with lemon wedges."
    },
    {
        "id": "recipe_006",
        "title": "Chicken Noodle Soup",
        "ingredients": "chicken breast, egg noodles, carrots, celery, onions, chicken broth, herbs, salt, pepper",
        "steps": "1. Sauté onions, carrots, and celery until soft. 2. Add chicken broth and bring to boil. 3. Add chicken and simmer until cooked. 4. Add noodles and cook until tender. 5. Season with herbs, salt, and pepper."
    },
    {
        "id": "recipe_007",
        "title": "Vegetarian Chili",
        "ingredients": "black beans, kidney beans, tomatoes, onions, bell peppers, chili powder, cumin, garlic, vegetable broth",
        "steps": "1. Sauté onions and peppers until soft. 2. Add garlic and spices, cook 1 minute. 3. Add beans, tomatoes, and broth. 4. Simmer for 30 minutes. 5. Season to taste and serve."
    },
    {
        "id": "recipe_008",
        "title": "Grilled Cheese Sandwich",
        "ingredients": "bread, cheddar cheese, butter, tomatoes",
        "steps": "1. Butter one side of each bread slice. 2. Add cheese and tomato slices between bread. 3. Cook in pan over medium heat until golden. 4. Flip and cook other side. 5. Serve hot."
    },
    {
        "id": "recipe_009",
        "title": "Caesar Salad",
        "ingredients": "romaine lettuce, parmesan cheese, croutons, caesar dressing, lemon, anchovies",
        "steps": "1. Wash and chop romaine lettuce. 2. Make caesar dressing with lemon and anchovies. 3. Toss lettuce with dressing. 4. Add croutons and parmesan. 5. Serve immediately."
    },
    {
        "id": "recipe_010",
        "title": "Chocolate Chip Cookies",
        "ingredients": "flour, butter, sugar, brown sugar, eggs, vanilla, chocolate chips, baking soda, salt",
        "steps": "1. Cream butter and sugars. 2. Add eggs and vanilla. 3. Mix in dry ingredients. 4. Fold in chocolate chips. 5. Bake at 375°F for 9-11 minutes."
    },
    {
        "id": "recipe_011",
        "title": "Beef and Broccoli",
        "ingredients": "beef strips, broccoli, soy sauce, garlic, ginger, cornstarch, vegetable oil, rice",
        "steps": "1. Marinate beef in soy sauce and cornstarch. 2. Stir fry beef until browned. 3. Add garlic and ginger. 4. Add broccoli and stir fry until tender. 5. Serve over rice."
    },
    {
        "id": "recipe_012",
        "title": "Caprese Salad",
        "ingredients": "fresh mozzarella, tomatoes, basil, olive oil, balsamic vinegar, salt, pepper",
        "steps": "1. Slice tomatoes and mozzarella. 2. Arrange on plate alternating slices. 3. Add fresh basil leaves. 4. Drizzle with olive oil and balsamic. 5. Season with salt and pepper."
    },
    {
        "id": "recipe_013",
        "title": "Pancakes",
        "ingredients": "flour, milk, eggs, butter, sugar, baking powder, salt, vanilla",
        "steps": "1. Mix dry ingredients in bowl. 2. Whisk wet ingredients separately. 3. Combine wet and dry ingredients. 4. Cook on griddle until bubbles form. 5. Flip and cook until golden."
    },
    {
        "id": "recipe_014",
        "title": "Vegetable Stir Fry",
        "ingredients": "mixed vegetables, soy sauce, garlic, ginger, sesame oil, vegetable oil, rice",
        "steps": "1. Heat oil in wok or large pan. 2. Add garlic and ginger, stir fry briefly. 3. Add vegetables and stir fry until tender-crisp. 4. Add soy sauce and sesame oil. 5. Serve over rice."
    },
    {
        "id": "recipe_015",
        "title": "Chicken Parmesan",
        "ingredients": "chicken breast, breadcrumbs, parmesan cheese, marinara sauce, mozzarella cheese, eggs, flour",
        "steps": "1. Pound chicken thin and season. 2. Dredge in flour, egg, then breadcrumbs. 3. Pan fry until golden and cooked through. 4. Top with marinara and mozzarella. 5. Broil until cheese melts."
    }
]

def get_embedded_recipes() -> List[Recipe]:
    """Get the embedded recipes as Recipe objects."""
    recipes = []
    for recipe_data in EMBEDDED_RECIPES_DATA:
        recipe = Recipe(
            id=recipe_data["id"],
            title=recipe_data["title"],
            ingredients=recipe_data["ingredients"],
            steps=recipe_data["steps"]
        )
        recipes.append(recipe)
    return recipes

def get_embedded_recipes_with_embeddings() -> List[Recipe]:
    """Get embedded recipes with pre-computed embeddings."""
    recipes = get_embedded_recipes()
    recipes_with_embeddings = []
    
    print(f"🔄 Generating embeddings for {len(recipes)} recipes...")
    
    for i, recipe in enumerate(recipes, 1):
        try:
            print(f"  Processing recipe {i}/{len(recipes)}: {recipe.title}")
            # Generate embedding for ingredients
            embedding = get_embedding(recipe.ingredients)
            if embedding and len(embedding) > 0:
                recipe.embedding = embedding
                print(f"    ✅ Generated embedding with {len(embedding)} dimensions")
            else:
                print(f"    ⚠️  Empty embedding generated")
                recipe.embedding = None
            recipes_with_embeddings.append(recipe)
        except Exception as e:
            print(f"    ❌ Error generating embedding for recipe {recipe.id}: {e}")
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"    ⚠️  Rate limit hit, skipping embedding generation")
            # Still add recipe without embedding
            recipe.embedding = None
            recipes_with_embeddings.append(recipe)
    
    successful_embeddings = sum(1 for r in recipes_with_embeddings if r.embedding is not None)
    print(f"✅ Successfully generated {successful_embeddings}/{len(recipes)} embeddings")
    
    if successful_embeddings == 0:
        print("⚠️  No embeddings generated - will use keyword matching only")
    
    return recipes_with_embeddings

def initialize_embedded_recipes_kb(kb_file: str = "recipe_knowledge_base.json") -> bool:
    """Initialize the knowledge base with embedded recipes."""
    try:
        import os
        from services.recipe_service import save_recipes_to_kb, load_recipes_from_kb
        
        # Check if knowledge base already exists and has embeddings
        if os.path.exists(kb_file):
            existing_recipes = load_recipes_from_kb(kb_file)
            if existing_recipes and len(existing_recipes) > 0:
                # Check if embeddings exist and are not null
                has_embeddings = any(
                    recipe.get("embedding") is not None and 
                    recipe.get("embedding") != [] and 
                    len(recipe.get("embedding", [])) > 0
                    for recipe in existing_recipes
                )
                if has_embeddings:
                    print(f"✅ Knowledge base already exists with {len(existing_recipes)} recipes and embeddings")
                    return True
                else:
                    print("🔄 Knowledge base exists but embeddings are missing, regenerating...")
        
        print("🔄 Initializing knowledge base with embedded recipes...")
        
        # Get recipes with embeddings
        recipes_with_embeddings = get_embedded_recipes_with_embeddings()
        
        # Save to knowledge base
        success = save_recipes_to_kb(recipes_with_embeddings, kb_file)
        
        if success:
            successful_embeddings = sum(1 for r in recipes_with_embeddings if r.embedding is not None)
            if successful_embeddings > 0:
                print(f"✅ Successfully initialized knowledge base with {len(recipes_with_embeddings)} recipes ({successful_embeddings} with embeddings)")
            else:
                print(f"✅ Successfully initialized knowledge base with {len(recipes_with_embeddings)} recipes (keyword search only)")
        else:
            print("❌ Failed to initialize knowledge base with embedded recipes")
            
        return success
        
    except Exception as e:
        print(f"❌ Error initializing embedded recipes: {e}")
        return False
