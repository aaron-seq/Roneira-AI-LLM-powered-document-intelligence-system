# ==============================================================================
# Free LLM Service Implementation
# Using DeepSeek API + Local Ollama for document intelligence
# ==============================================================================

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import aiohttp
import httpx
from langchain.llms import Ollama
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for free LLM services"""
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "deepseek-coder:6.7b"
    fallback_model: str = "llama3.1:8b"
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout: int = 60

class FreeLLMService:
    """Free LLM service supporting DeepSeek API and local Ollama"""
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or self.config.deepseek_api_key
        self.use_local = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
        
        # Initialize services
        self._init_services()
    
    def _init_services(self):
        """Initialize LLM services"""
        try:
            # Local Ollama setup
            if self.use_local:
                self.ollama_client = Ollama(
                    base_url=self.config.ollama_base_url,
                    model=self.config.local_model,
                    temperature=self.config.temperature
                )
                logger.info(f"Initialized Ollama with model: {self.config.local_model}")
            
            # DeepSeek API setup
            if self.deepseek_api_key:
                self.deepseek_headers = {
                    "Authorization": f"Bearer {self.deepseek_api_key}",
                    "Content-Type": "application/json"
                }
                logger.info("Initialized DeepSeek API client")
            else:
                logger.warning("No DeepSeek API key provided")
                
        except Exception as e:
            logger.error(f"Error initializing LLM services: {e}")
    
    async def _call_deepseek_api(self, prompt: str, **kwargs) -> str:
        """Call DeepSeek API"""
        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API key not configured")
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                response = await client.post(
                    f"{self.config.deepseek_base_url}/chat/completions",
                    headers=self.deepseek_headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except httpx.HTTPError as e:
                logger.error(f"DeepSeek API error: {e}")
                raise
    
    async def _call_ollama_local(self, prompt: str, **kwargs) -> str:
        """Call local Ollama"""
        if not hasattr(self, 'ollama_client'):
            raise ValueError("Ollama client not initialized")
        
        try:
            # Use async wrapper for Ollama
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.ollama_client.invoke(prompt)
            )
            return result
            
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            # Try fallback model
            try:
                fallback_client = Ollama(
                    base_url=self.config.ollama_base_url,
                    model=self.config.fallback_model
                )
                result = await loop.run_in_executor(
                    None,
                    lambda: fallback_client.invoke(prompt)
                )
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback model error: {fallback_error}")
                raise
    
    async def generate_response(self, prompt: str, use_cloud: bool = None, **kwargs) -> str:
        """Generate response using available LLM services"""
        
        # Determine which service to use
        if use_cloud is None:
            use_cloud = not self.use_local or not hasattr(self, 'ollama_client')
        
        try:
            if use_cloud and self.deepseek_api_key:
                logger.info("Using DeepSeek API")
                return await self._call_deepseek_api(prompt, **kwargs)
            
            elif self.use_local and hasattr(self, 'ollama_client'):
                logger.info("Using local Ollama")
                return await self._call_ollama_local(prompt, **kwargs)
            
            else:
                raise ValueError("No LLM service available")
                
        except Exception as e:
            logger.error(f"Primary LLM service failed: {e}")
            
            # Try fallback service
            try:
                if not use_cloud and self.deepseek_api_key:
                    logger.info("Falling back to DeepSeek API")
                    return await self._call_deepseek_api(prompt, **kwargs)
                elif use_cloud and hasattr(self, 'ollama_client'):
                    logger.info("Falling back to local Ollama")
                    return await self._call_ollama_local(prompt, **kwargs)
                else:
                    raise e
            except Exception as fallback_error:
                logger.error(f"Fallback service also failed: {fallback_error}")
                raise fallback_error
    
    async def analyze_document(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze document content using free LLM"""
        
        prompts = {
            "general": f"""
Analyze the following document and provide:
1. Summary (2-3 sentences)
2. Key topics or themes
3. Important entities (names, dates, locations)
4. Document type classification

Document text:
{text[:3000]}...

Provide your analysis in JSON format.
""",
            "extraction": f"""
Extract structured information from this document:
1. Key-value pairs
2. Tables and lists
3. Important dates and numbers
4. Names and entities

Document text:
{text[:3000]}...

Return as structured JSON.
""",
            "summary": f"""
Provide a comprehensive summary of this document including:
- Main purpose/topic
- Key findings or information
- Action items (if any)
- Important details

Document text:
{text[:4000]}...
"""
        }
        
        prompt = prompts.get(analysis_type, prompts["general"])
        
        try:
            response = await self.generate_response(prompt, max_tokens=1024)
            
            # Try to parse as JSON, fallback to text
            try:
                import json
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "analysis_type": analysis_type,
                    "content": response,
                    "raw_text": text[:500] + "..." if len(text) > 500 else text
                }
                
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {
                "error": str(e),
                "analysis_type": analysis_type,
                "status": "failed"
            }
    
    async def generate_insights(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from document data"""
        
        prompt = f"""
Based on the following document analysis, provide actionable insights:

Document Data:
{str(document_data)[:2000]}...

Provide insights including:
1. Key takeaways
2. Potential issues or concerns
3. Recommendations or next steps
4. Related topics to explore

Format as JSON with clear categories.
"""
        
        try:
            response = await self.generate_response(prompt, max_tokens=800)
            
            try:
                import json
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "insights": response,
                    "source_data": document_data
                }
                
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of LLM services"""
        health = {
            "deepseek_api": False,
            "ollama_local": False,
            "services_available": []
        }
        
        # Check DeepSeek API
        if self.deepseek_api_key:
            try:
                test_response = await self._call_deepseek_api("Hello", max_tokens=10)
                if test_response:
                    health["deepseek_api"] = True
                    health["services_available"].append("deepseek")
            except Exception as e:
                logger.warning(f"DeepSeek API health check failed: {e}")
        
        # Check Ollama
        if hasattr(self, 'ollama_client'):
            try:
                test_response = await self._call_ollama_local("Hello")
                if test_response:
                    health["ollama_local"] = True
                    health["services_available"].append("ollama")
            except Exception as e:
                logger.warning(f"Ollama health check failed: {e}")
        
        health["status"] = "healthy" if health["services_available"] else "unhealthy"
        return health

# Initialize global service instance
_llm_service = None

def get_llm_service() -> FreeLLMService:
    """Get or create LLM service instance"""
    global _llm_service
    if _llm_service is None:
        config = LLMConfig(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            local_model=os.getenv("LOCAL_MODEL", "deepseek-coder:6.7b")
        )
        _llm_service = FreeLLMService(config)
    return _llm_service