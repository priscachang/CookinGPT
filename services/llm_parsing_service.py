import os
import re
from typing import List, Tuple
from mistralai import Mistral  # pip install mistralai

class LLMParsingService:
    """Service for parsing user input using Mistral AI API to separate ingredients and preferences."""
    
    def __init__(self, model: str = "mistral-small-latest"):
        # Use Mistral API key instead of OpenAI
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.client = None
        if self.api_key:
            try:
                # Mistral client initialization
                self.client = Mistral(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Mistral client: {e}")
                self.client = None
        
        self.parsing_prompt = """The user will describe what they have in the fridge and their cooking needs.  
You must respond with exactly two lines in this format:

Output 1: ingredient1, ingredient2, ingredient3  
Output 2: preference1, preference2, preference3

Rules:
- Output 1: list only the normalized ingredients the user already has, comma-separated, no quantities.  
- Output 2: list all other meaningful requirements (time limit, servings, dietary restrictions, cuisine preferences, equipment, etc.), comma-separated.  
- If a field is not provided by the user, omit it entirely.  
- Do not add extra text, explanations, or formatting."""
    
    def parse_user_input(self, user_input: str) -> Tuple[List[str], List[str]]:
        """
        Parse user input to separate ingredients and preferences.
        
        Args:
            user_input: Raw user input describing ingredients and preferences
            
        Returns:
            Tuple of (ingredients_list, preferences_list)
        """
        if not self.client:
            print("Mistral client not available, using fallback parsing")
            return self._fallback_parsing(user_input)
        
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.parsing_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            # Mistral returns choices -> message -> content (same access pattern)
            content = response.choices[0].message.content.strip()
            ingredients, preferences = self._parse_llm_response(content)
            return ingredients, preferences

        except Exception as e:
            print(f"Error in LLM parsing (Mistral): {e}")
            return self._fallback_parsing(user_input)
    
    def _parse_llm_response(self, response: str) -> Tuple[List[str], List[str]]:
        """
        Parse the LLM response to extract ingredients and preferences.
        """
        lines = response.strip().split('\n')
        ingredients: List[str] = []
        preferences: List[str] = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("Output 1:"):
                ingredient_text = line.replace("Output 1:", "").strip()
                if ingredient_text:
                    ingredients = [ing.strip() for ing in ingredient_text.split(',') if ing.strip()]
            elif line.startswith("Output 2:"):
                preference_text = line.replace("Output 2:", "").strip()
                if preference_text:
                    preferences = [pref.strip() for pref in preference_text.split(',') if pref.strip()]
        return ingredients, preferences
    
    def _fallback_parsing(self, user_input: str) -> Tuple[List[str], List[str]]:
        """
        Fallback parsing method when LLM fails.
        """
        ingredients: List[str] = []
        preferences: List[str] = []
        
        ingredient_patterns = [
            r'i have\s+([^.]+)',
            r'i\'ve got\s+([^.]+)',
            r'ingredients?\s*:?\s*([^.]+)',
            r'available\s+([^.]+)',
        ]
        preference_patterns = [
            r'i want\s+([^.]+)',
            r'i need\s+([^.]+)',
            r'something\s+([^.]+)',
            r'for\s+([^.]+)',
            r'servings?\s*:?\s*([^.]+)',
            r'quick\s+([^.]+)',
        ]
        
        user_input_lower = user_input.lower()
        for pattern in ingredient_patterns:
            matches = re.findall(pattern, user_input_lower)
            for match in matches:
                parts = re.split(r'[,;]\s*', match.strip())
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 1:
                        ingredients.append(part)
        for pattern in preference_patterns:
            matches = re.findall(pattern, user_input_lower)
            for match in matches:
                parts = re.split(r'[,;]\s*', match.strip())
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 1:
                        preferences.append(part)
        
        if not ingredients:
            parts = re.split(r'[,;]\s*', user_input)
            for part in parts:
                part = part.strip()
                if part and len(part) > 1:
                    ingredients.append(part)
        
        ingredients = list(dict.fromkeys(ingredients))
        preferences = list(dict.fromkeys(preferences))
        return ingredients, preferences

# Create a singleton instance
llm_parsing_service = LLMParsingService()
