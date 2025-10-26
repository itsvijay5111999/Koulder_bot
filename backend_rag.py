"""
RAG Knowledge Base for Research Papers
Stores and retrieves research papers using Pinecone and Groq LLM
"""

import os
import json
import datetime
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
from typing import List, Dict, Optional
import hashlib

# Pinecone for Vector DB
from pinecone import Pinecone, ServerlessSpec

# Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer

# Groq for LLM
from groq import Groq

# Configuration
PINECONE_INDEX_NAME = "research-papers"
PINECONE_DIMENSION = 384  # all-MiniLM-L6-v2 dimension
GROQ_MODEL = "llama-3.3-70b-versatile"

class ResearchPaperRAGPinecone:
    """RAG system for research papers using Pinecone and Groq"""
    
    def __init__(self, groq_api_key: str, pinecone_api_key: str, pinecone_environment: str = "us-east-1"):
        """
        Initialize the RAG system with Pinecone
        
        Args:
            groq_api_key: Groq API key for LLM
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment (e.g., 'us-east-1')
        """
        self.groq_api_key = groq_api_key
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize embedding model (same as ChromaDB default)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Create or connect to index
        self._setup_index(pinecone_environment)
    
    def _setup_index(self, environment: str):
        """Create or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx['name'] for idx in existing_indexes]
            
            if PINECONE_INDEX_NAME not in index_names:
                print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
                
                # Create index with serverless spec (free tier)
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=environment
                    )
                )
                print(f"✅ Created new index: {PINECONE_INDEX_NAME}")
            else:
                print(f"✅ Connected to existing index: {PINECONE_INDEX_NAME}")
            
            # Connect to index
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            
            # Get stats
            stats = self.index.describe_index_stats()
            print(f"✅ Index has {stats['total_vector_count']} vectors")
            
        except Exception as e:
            print(f"Error setting up Pinecone index: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using sentence transformers"""
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def _create_paper_id(self, paper: Dict) -> str:
        """Create a unique ID for a paper"""
        # Use arxiv_id if available, otherwise hash the title
        if paper.get('id'):
            return f"paper_{paper['id']}"
        else:
            title_hash = hashlib.md5(paper['title'].encode()).hexdigest()
            return f"paper_{title_hash}"
    
    def fetch_arxiv_papers(self, category: str = "cs.AI", max_results: int = 50) -> List[Dict]:
        """Fetch papers from arXiv API"""
        base_url = "http://export.arxiv.org/api/query?"
        search_query = f"cat:{category}"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        query_url = base_url + "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
        
        try:
            response = requests.get(query_url, timeout=30)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                papers = []
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    paper = {
                        'id': entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1],
                        'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' '),
                        'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().replace('\n', ' '),
                        'authors': [author.find('{http://www.w3.org/2005/Atom}name').text 
                                   for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
                        'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
                        'categories': [],
                        'arxiv_url': entry.find('{http://www.w3.org/2005/Atom}id').text,
                        'source': 'arXiv'
                    }
                    
                    # Get categories
                    for cat in entry.findall('{http://arxiv.org/schemas/atom}primary_category'):
                        paper['categories'].append(cat.get('term'))
                    
                    papers.append(paper)
                
                return papers
        except Exception as e:
            print(f"Error fetching papers: {e}")
            return []
    
    def fetch_huggingface_papers(self, max_results: int = 50) -> List[Dict]:
        """Fetch curated papers from Hugging Face Daily Papers"""
        try:
            url = "https://huggingface.co/api/daily_papers"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                papers = []
                for item in data[:max_results]:
                    arxiv_id = item.get('paper', {}).get('arxivId', '')
                    paper = {
                        'id': arxiv_id,
                        'title': item.get('paper', {}).get('title', 'No Title'),
                        'summary': item.get('paper', {}).get('summary', 'No summary available.'),
                        'authors': [author.get('name', '') for author in item.get('paper', {}).get('authors', [])],
                        'published': item.get('publishedAt', ''),
                        'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}",
                        'categories': ['AI/ML', 'Trending'],
                        'source': 'HuggingFace',
                        'upvotes': item.get('paper', {}).get('upvotes', 0)
                    }
                    papers.append(paper)
                
                print(f"✅ Fetched {len(papers)} papers from Hugging Face")
                return papers
            else:
                print(f"Error fetching HF papers: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching Hugging Face papers: {e}")
            return []
    
    def add_papers_to_db(self, papers: List[Dict]) -> int:
        """Add papers to Pinecone"""
        if not papers:
            return 0
        
        vectors_to_upsert = []
        added_count = 0
        
        for paper in papers:
            paper_id = self._create_paper_id(paper)
            
            # Check if paper already exists
            try:
                existing = self.index.fetch(ids=[paper_id])
                if existing['vectors']:
                    continue  # Skip if already exists
            except:
                pass
            
            # Create document text (title + summary for better retrieval)
            doc_text = f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"
            
            # Generate embedding
            embedding = self._generate_embedding(doc_text)
            
            # Prepare metadata (Pinecone has metadata size limits)
            metadata = {
                'title': paper['title'][:500],
                'authors': ', '.join(paper['authors'][:5])[:500],
                'published': paper['published'][:10],
                'categories': ', '.join(paper['categories'])[:200],
                'arxiv_url': paper['arxiv_url'][:500],
                'source': paper.get('source', 'arXiv'),
                'added_date': datetime.datetime.now().strftime("%Y-%m-%d"),
                'full_text': doc_text[:1000]  # Store snippet for retrieval
            }
            
            vectors_to_upsert.append({
                'id': paper_id,
                'values': embedding,
                'metadata': metadata
            })
            added_count += 1
        
        # Upsert in batches of 100 (Pinecone limit)
        if vectors_to_upsert:
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                self.index.upsert(vectors=batch)
            
            print(f"✅ Added {added_count} new papers to Pinecone")
            return added_count
        
        return 0
    
    def search_papers(self, query: str, n_results: int = 5) -> Dict:
        """Search for relevant papers using semantic search"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=n_results,
                include_metadata=True
            )
            
            if not results['matches']:
                return {
                    'success': False,
                    'message': 'No relevant papers found',
                    'papers': []
                }
            
            papers = []
            for match in results['matches']:
                metadata = match['metadata']
                paper_data = {
                    'id': match['id'],
                    'title': metadata.get('title', 'Unknown'),
                    'authors': metadata.get('authors', 'Unknown'),
                    'published': metadata.get('published', 'Unknown'),
                    'categories': metadata.get('categories', 'Unknown'),
                    'arxiv_url': metadata.get('arxiv_url', '#'),
                    'source': metadata.get('source', 'Unknown'),
                    'content': metadata.get('full_text', ''),
                    'score': match['score']
                }
                papers.append(paper_data)
            
            return {
                'success': True,
                'papers': papers,
                'count': len(papers)
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Search error: {str(e)}',
                'papers': []
            }
    
    def answer_question(self, question: str, n_results: int = 5) -> Dict:
        """Answer a question using RAG"""
        # Search for relevant papers
        search_results = self.search_papers(question, n_results)
        
        if not search_results['success'] or not search_results['papers']:
            return {
                'success': False,
                'answer': 'I could not find any relevant research papers to answer your question. Try updating the database with latest papers.',
                'sources': []
            }
        
        # Prepare context from retrieved papers
        context_parts = []
        sources = []
        
        for i, paper in enumerate(search_results['papers'], 1):
            context_parts.append(f"Paper {i}:\n{paper['content']}\n")
            sources.append({
                'title': paper['title'],
                'authors': paper['authors'],
                'url': paper['arxiv_url'],
                'published': paper['published'],
                'source': paper.get('source', 'Unknown')
            })
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt for Groq
        prompt = f"""You are an AI research assistant. Answer the following question based on the research papers provided below.

Question: {question}

Research Papers Context:
{context}

Instructions:
- Provide a clear, comprehensive answer based on the papers above
- Reference specific papers when making claims (e.g., "According to Paper 1...")
- If the papers don't fully answer the question, say so
- Be accurate and cite the papers appropriately
- Keep your answer concise but informative

Answer:"""
        
        try:
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI research assistant that answers questions based on academic papers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'papers_used': len(sources)
            }
        
        except Exception as e:
            return {
                'success': False,
                'answer': f'Error generating answer: {str(e)}',
                'sources': sources
            }
    
    def update_daily_papers(self, categories: List[str] = None, include_huggingface: bool = True) -> Dict:
        """Update database with latest papers from arXiv and Hugging Face"""
        if categories is None:
            categories = ["cs.AI", "cs.LG", "cs.CL"]
        
        total_added = 0
        results = {}
        all_papers = []
        
        # Fetch from Hugging Face
        if include_huggingface:
            print("Fetching papers from Hugging Face...")
            hf_papers = self.fetch_huggingface_papers(max_results=30)
            all_papers.extend(hf_papers)
            results['huggingface'] = {
                'fetched': len(hf_papers),
                'added': 0
            }
        
        # Fetch from arXiv
        for category in categories:
            print(f"Fetching papers from arXiv ({category})...")
            papers = self.fetch_arxiv_papers(category=category, max_results=30)
            all_papers.extend(papers)
            results[category] = {
                'fetched': len(papers),
                'added': 0
            }
        
        # Remove duplicates based on ID or title
        seen = set()
        unique_papers = []
        for paper in all_papers:
            identifier = paper.get('id') or paper.get('title')
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique_papers.append(paper)
        
        # Add to database
        added = self.add_papers_to_db(unique_papers)
        total_added = added
        
        # Update results
        if include_huggingface and 'huggingface' in results:
            hf_added = sum(1 for p in unique_papers if p.get('source') == 'HuggingFace')
            results['huggingface']['added'] = hf_added
        
        for category in categories:
            if category in results:
                cat_added = sum(1 for p in unique_papers if category in p.get('categories', []))
                results[category]['added'] = cat_added
        
        return {
            'success': True,
            'total_added': total_added,
            'total_fetched': len(all_papers),
            'unique_papers': len(unique_papers),
            'categories': results,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'success': True,
                'total_papers': stats['total_vector_count'],
                'index_name': PINECONE_INDEX_NAME,
                'dimension': PINECONE_DIMENSION
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_database(self):
        """Clear all papers from the database"""
        try:
            # Delete all vectors in the index
            self.index.delete(delete_all=True)
            return {'success': True, 'message': 'Database cleared successfully'}
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Convenience functions for easy import
def initialize_rag(groq_api_key: str, pinecone_api_key: str) -> ResearchPaperRAGPinecone:
    """Initialize the RAG system with Pinecone"""
    return ResearchPaperRAGPinecone(groq_api_key, pinecone_api_key)


# Example usage
if __name__ == "__main__":
    # Initialize with your API keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_pinecone_api_key_here")
    
    rag = ResearchPaperRAGPinecone(GROQ_API_KEY, PINECONE_API_KEY)
    
    # Update database with latest papers
    print("\nUpdating database with latest papers...")
    update_result = rag.update_daily_papers(
        categories=["cs.AI", "cs.LG", "cs.CL"],
        include_huggingface=True
    )
    print(f"Added {update_result['total_added']} new papers")
    
    # Get stats
    stats = rag.get_stats()
    print(f"\nDatabase has {stats['total_papers']} papers")
    
    # Ask a question
    question = "What are the latest developments in large language models?"
    print(f"\nQuestion: {question}")
    result = rag.answer_question(question)
    
    if result['success']:
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nBased on {result['papers_used']} papers")
    else:
        print(f"Error: {result['answer']}")