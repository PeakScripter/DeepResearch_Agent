import os
import arxiv
import requests
import logging
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json
import re
from urllib.parse import quote

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
class ResearchState(TypedDict):
    query: str
    initial_results: List[Dict]
    refined_queries: List[str]
    cross_references: List[Dict]
    synthesis: str
    follow_up_questions: List[str]
    current_step: str
    research_depth: int
    max_depth: int
    all_findings: List[Dict]
    research_summary: str
    recommendations: List[str]
    all_papers: List[Dict]
    formatted_references: str

class CitationFormatter:
    
    @staticmethod
    def format_apa(paper: Dict) -> str:
        authors = paper.get('authors', 'N/A')
        title = paper.get('title', 'N/A')
        published = paper.get('published', 'N/A')
        url = paper.get('url', '')
        arxiv_id = paper.get('arxiv_id', '')
        
        if authors != 'N/A':
            author_list = [author.strip() for author in authors.split(',')]
            if len(author_list) > 3:
                formatted_authors = ', '.join(author_list[:3]) + ', et al.'
            else:
                formatted_authors = ', '.join(author_list)
        else:
            formatted_authors = 'Unknown'
        
        year = published.split('-')[0] if published != 'N/A' and '-' in published else 'N/A'
        
        formatted_title = title.replace('"', '').strip()
        if not formatted_title.endswith('.'):
            formatted_title += '.'
        citation = f"{formatted_authors} ({year}). {formatted_title}"
        
        if arxiv_id and arxiv_id != 'N/A':
            citation += f" arXiv preprint arXiv:{arxiv_id}"
        
        if url:
            citation += f" Retrieved from {url}"
        
        return citation
    
    @staticmethod
    def format_ieee(paper: Dict) -> str:
        authors = paper.get('authors', 'N/A')
        title = paper.get('title', 'N/A')
        published = paper.get('published', 'N/A')
        arxiv_id = paper.get('arxiv_id', '')
        
        if authors != 'N/A':
            author_list = [author.strip() for author in authors.split(',')]
            if len(author_list) > 6:
                formatted_authors = ', '.join(author_list[:6]) + ', et al.'
            else:
                formatted_authors = ', '.join(author_list)
        else:
            formatted_authors = 'Unknown'
        
        year = published.split('-')[0] if published != 'N/A' and '-' in published else 'N/A'
        
        formatted_title = title.replace('"', '').strip()
        citation = f"{formatted_authors}, \"{formatted_title},\""
        
        if arxiv_id and arxiv_id != 'N/A':
            citation += f" arXiv preprint arXiv:{arxiv_id}"
        else:
            citation += f" {year}"
        
        return citation
    
    @staticmethod
    def extract_doi(paper: Dict) -> Optional[str]:
        return None

class ArxivSearchTool:
    
    def search(self, query: str, max_results: int = 5, sort_by: str = "relevance") -> str:
        try:
            sort_mapping = {
                "relevance": arxiv.SortCriterion.Relevance,
                "submitted_date": arxiv.SortCriterion.SubmittedDate,
                "last_updated": arxiv.SortCriterion.LastUpdatedDate
            }
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_mapping.get(sort_by, arxiv.SortCriterion.Relevance)
            )
            
            results_summary = []
            client = arxiv.Client()
            for i, result in enumerate(client.results(search), 1):
                pub_date = result.published.strftime('%Y-%m-%d') if result.published else 'N/A'
                
                authors = ', '.join([author.name for author in result.authors]) if result.authors else 'N/A'
                
                abstract = result.summary.replace('\n', ' ').strip() if result.summary else 'N/A'
                
                arxiv_id = 'N/A'
                if result.entry_id:
                    arxiv_id = result.entry_id.split('/')[-1]
                elif hasattr(result, 'arxiv_id'):
                    arxiv_id = result.arxiv_id
                
                categories = ', '.join(result.categories) if hasattr(result, 'categories') and result.categories else 'N/A'
                
                journal_ref = getattr(result, 'journal_ref', None) or 'N/A'
                
                result_info = {
                    "title": result.title,
                    "authors": authors,
                    "published": pub_date,
                    "abstract": abstract,
                    "url": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "arxiv_id": arxiv_id,
                    "categories": categories,
                    "journal_ref": journal_ref,
                    "doi": CitationFormatter.extract_doi({}),
                    "source": "arXiv"
                }
                
                results_summary.append(result_info)
            
            if not results_summary:
                return f"No results found on arXiv for query: '{query}'"
            
            return json.dumps(results_summary, indent=2)
            
        except Exception as e:
            logger.error(f"Error during arXiv search for '{query}': {e}")
            return f"Error performing arXiv search: {e}"

class PapersWithCodeSearchTool:
    
    def search(self, query: str, max_results: int = 5) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(
                "https://paperswithcode.com/api/v1/search/",
                params={
                    'q': query,
                    'items_per_page': max_results,
                    'ordering': 'relevance'
                },
                headers=headers,
                timeout=15
            )
            
            logger.info(f"PapersWithCode response status: {response.status_code}")
            logger.info(f"PapersWithCode response headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                logger.warning(f"PapersWithCode returned status {response.status_code}")
                return json.dumps([], indent=2)
            
            content = response.text.strip()
            if not content:
                logger.warning("PapersWithCode returned empty response")
                return json.dumps([], indent=2)
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from PapersWithCode: {e}")
                logger.error(f"Response content: {content[:500]}...")
                return json.dumps([], indent=2)

            results_summary = []
            papers = data.get('results', [])
            
            logger.info(f"Found {len(papers)} papers from PapersWithCode")
            
            for paper in papers:
                paper_data = paper.get('paper', {})
                repositories = paper.get('repositories', [])
                
                repo_info = []
                for repo in repositories[:3]:
                    repo_info.append({
                        "name": repo.get('name', 'N/A'),
                        "url": repo.get('url', 'N/A'),
                        "stars": repo.get('stars', 0),
                        "framework": repo.get('framework', 'N/A')
                    })
                
                published_date = paper_data.get('published', 'N/A')
                authors = paper_data.get('authors', [])
                author_names = ', '.join([author.get('name', '') for author in authors]) if authors else 'N/A'
                
                result_info = {
                    "title": paper_data.get('title', 'N/A'),
                    "authors": author_names,
                    "published": published_date,
                    "arxiv_id": paper_data.get('arxiv_id', 'N/A'),
                    "url": paper_data.get('url_abs', 'N/A'),
                    "repositories": repo_info,
                    "repo_count": len(repositories),
                    "abstract": paper_data.get('abstract', 'N/A'),
                    "categories": paper_data.get('categories', 'N/A'),
                    "source": "PapersWithCode"
                }
                
                results_summary.append(result_info)

            logger.info(f"Processed {len(results_summary)} papers from PapersWithCode")
            return json.dumps(results_summary, indent=2)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during PapersWithCode search for '{query}': {e}")
            return json.dumps([], indent=2)
        except Exception as e:
            logger.error(f"Unexpected error during PapersWithCode search for '{query}': {e}")
            return json.dumps([], indent=2)

class GoogleScholarSearchTool:
    
    def search(self, query: str, max_results: int = 5) -> str:
        try:
            search_suggestions = [
                f"Google Scholar search for: {query}",
                "Note: Implement proper Google Scholar API integration for full functionality",
                f"Expected results: {max_results} papers"
            ]
            
            return json.dumps({
                "search_query": query,
                "max_results": max_results,
                "note": "Google Scholar integration needs proper API setup",
                "suggestions": search_suggestions
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error during Google Scholar search for '{query}': {e}")
            return f"Error performing Google Scholar search: {e}"

class DeepResearchAssistant:
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.arxiv_tool = ArxivSearchTool()
        self.pwc_tool = PapersWithCodeSearchTool()
        self.scholar_tool = GoogleScholarSearchTool()
        self.citation_formatter = CitationFormatter()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ResearchState)
        
        workflow.add_node("initial_search", self._initial_search)
        workflow.add_node("analyze_initial_results", self._analyze_initial_results)
        workflow.add_node("refined_search", self._refined_search)
        workflow.add_node("cross_reference_search", self._cross_reference_search)
        workflow.add_node("collect_all_papers", self._collect_all_papers)
        workflow.add_node("synthesize_findings", self._synthesize_findings)
        workflow.add_node("generate_follow_up", self._generate_follow_up)
        workflow.add_node("final_synthesis", self._final_synthesis)
        
        workflow.set_entry_point("initial_search")
        
        workflow.add_edge("initial_search", "analyze_initial_results")
        workflow.add_edge("analyze_initial_results", "refined_search")
        workflow.add_edge("refined_search", "cross_reference_search")
        workflow.add_edge("cross_reference_search", "collect_all_papers")
        workflow.add_edge("collect_all_papers", "synthesize_findings")
        workflow.add_edge("synthesize_findings", "generate_follow_up")
        workflow.add_edge("generate_follow_up", "final_synthesis")
        workflow.add_edge("final_synthesis", END)
        
        return workflow.compile()
    
    def _initial_search(self, state: ResearchState) -> ResearchState:
        logger.info(f"Starting initial search for: {state['query']}")
        
        arxiv_results = self.arxiv_tool.search(state['query'], max_results=5)
        
        pwc_results = self.pwc_tool.search(state['query'], max_results=5)
        
        scholar_results = self.scholar_tool.search(state['query'], max_results=5)
        
        initial_results = {
            "arxiv": json.loads(arxiv_results) if not arxiv_results.startswith("Error") else [],
            "papers_with_code": json.loads(pwc_results) if not pwc_results.startswith("Error") else [],
            "google_scholar": json.loads(scholar_results) if not scholar_results.startswith("Error") else []
        }
        
        state["initial_results"] = [initial_results]
        state["current_step"] = "initial_search_completed"
        state["research_depth"] = 1
        
        logger.info(f"Initial search completed. Found {len(initial_results['arxiv'])} arXiv papers, {len(initial_results['papers_with_code'])} PapersWithCode results")
        
        return state
    
    def _analyze_initial_results(self, state: ResearchState) -> ResearchState:
        logger.info("Analyzing initial results and generating refined queries")
        
        initial_results = state["initial_results"][0]
        analysis_prompt = f"""
        Analyze these initial research results and generate 3-5 refined search queries to deepen the research.
        
        Original Query: {state['query']}
        
        Initial Results:
        {json.dumps(initial_results, indent=2)}
        
        Generate refined queries that:
        1. Focus on specific aspects or gaps identified in the initial results
        2. Look for recent developments (last 2 years)
        3. Search for related methodologies or applications
        4. Find papers that cite or are cited by the most relevant initial papers
        
        Return only a JSON array of refined query strings.
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            refined_queries = json.loads(response.text)
            
            if isinstance(refined_queries, str):
                refined_queries = [refined_queries]
            
            state["refined_queries"] = refined_queries[:5]
            state["current_step"] = "analysis_completed"
            
            logger.info(f"Generated {len(state['refined_queries'])} refined queries")
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            state["refined_queries"] = [state['query'] + " recent developments"]
        
        return state
    
    def _refined_search(self, state: ResearchState) -> ResearchState:
        logger.info("Performing refined searches")
        
        refined_results = []
        
        for query in state["refined_queries"]:
            arxiv_results = self.arxiv_tool.search(query, max_results=3)
            
            pwc_results = self.pwc_tool.search(query, max_results=3)
            
            refined_result = {
                "query": query,
                "arxiv": json.loads(arxiv_results) if not arxiv_results.startswith("Error") else [],
                "papers_with_code": json.loads(pwc_results) if not pwc_results.startswith("Error") else []
            }
            
            refined_results.append(refined_result)
        
        state["initial_results"].extend(refined_results)
        state["current_step"] = "refined_search_completed"
        state["research_depth"] = 2
        
        logger.info(f"Refined search completed. Added {len(refined_results)} result sets")
        
        return state
    
    def _cross_reference_search(self, state: ResearchState) -> ResearchState:
        logger.info("Performing cross-reference search")
        
        arxiv_ids = []
        for result_set in state["initial_results"]:
            if "arxiv" in result_set:
                for paper in result_set["arxiv"]:
                    if isinstance(paper, dict) and "arxiv_id" in paper:
                        arxiv_ids.append(paper["arxiv_id"])
        
        arxiv_ids = list(set(arxiv_ids))[:3]
        
        cross_references = []
        
        for arxiv_id in arxiv_ids:
            citing_query = f"cites:{arxiv_id}"
            citing_results = self.arxiv_tool.search(citing_query, max_results=3)
            
            cited_query = f"cited_by:{arxiv_id}"
            cited_results = self.arxiv_tool.search(cited_query, max_results=3)
            try:
                citing_papers = json.loads(citing_results) if not citing_results.startswith("Error") and citing_results != "No results found on arXiv for query: 'cites:" + arxiv_id + "'" else []
            except:
                citing_papers = []
                
            try:
                cited_papers = json.loads(cited_results) if not cited_results.startswith("Error") and cited_results != "No results found on arXiv for query: 'cited_by:" + arxiv_id + "'" else []
            except:
                cited_papers = []
            
            cross_ref = {
                "original_paper": arxiv_id,
                "citing_papers": citing_papers,
                "cited_papers": cited_papers
            }
            
            cross_references.append(cross_ref)
        
        state["cross_references"] = cross_references
        state["current_step"] = "cross_reference_completed"
        state["research_depth"] = 3
        
        logger.info(f"Cross-reference search completed. Found references for {len(arxiv_ids)} papers")
        
        return state
    
    def _collect_all_papers(self, state: ResearchState) -> ResearchState:
        logger.info("Collecting all papers and formatting references")
        
        all_papers = []
        
        for result_set in state["initial_results"]:
            if "arxiv" in result_set:
                for paper in result_set["arxiv"]:
                    if isinstance(paper, dict) and paper.get("title") != "N/A":
                        all_papers.append(paper)
            
            if "papers_with_code" in result_set:
                for paper in result_set["papers_with_code"]:
                    if isinstance(paper, dict) and paper.get("title") != "N/A":
                        all_papers.append(paper)
        
        for cross_ref in state.get("cross_references", []):
            for paper in cross_ref.get("citing_papers", []):
                if isinstance(paper, dict) and paper.get("title") != "N/A":
                    all_papers.append(paper)
            for paper in cross_ref.get("cited_papers", []):
                if isinstance(paper, dict) and paper.get("title") != "N/A":
                    all_papers.append(paper)
        
        unique_papers = []
        seen_titles = set()
        seen_arxiv_ids = set()
        
        for paper in all_papers:
            title = paper.get("title", "").lower().strip()
            arxiv_id = paper.get("arxiv_id", "")
            
            if title in seen_titles or (arxiv_id and arxiv_id in seen_arxiv_ids):
                continue
            
            if title:
                seen_titles.add(title)
            if arxiv_id:
                seen_arxiv_ids.add(arxiv_id)
            
            unique_papers.append(paper)
        
        def sort_key(paper):
            published = paper.get("published", "N/A")
            if published == "N/A":
                return "0000-00-00"
            return published
        
        unique_papers.sort(key=sort_key, reverse=True)
        
        apa_references = []
        ieee_references = []
        
        for paper in unique_papers:
            try:
                apa_ref = self.citation_formatter.format_apa(paper)
                ieee_ref = self.citation_formatter.format_ieee(paper)
                
                if apa_ref and apa_ref not in apa_references:
                    apa_references.append(apa_ref)
                if ieee_ref and ieee_ref not in ieee_references:
                    ieee_references.append(ieee_ref)
            except Exception as e:
                logger.warning(f"Error formatting citation for paper: {e}")
                continue
        references_text = f"""
## References

### APA Style References:
{chr(10).join([f"{i+1}. {ref}" for i, ref in enumerate(apa_references)])}

### IEEE Style References:
{chr(10).join([f"[{i+1}] {ref}" for i, ref in enumerate(ieee_references)])}

### Summary:
- Total unique papers found: {len(unique_papers)}
- arXiv papers: {len([p for p in unique_papers if p.get('source') == 'arXiv'])}
- PapersWithCode papers: {len([p for p in unique_papers if p.get('source') == 'PapersWithCode'])}
"""
        
        state["all_papers"] = unique_papers
        state["formatted_references"] = references_text
        state["current_step"] = "papers_collected"
        
        logger.info(f"Collected {len(unique_papers)} unique papers and formatted references")
        
        return state
    
    def _synthesize_findings(self, state: ResearchState) -> ResearchState:
        logger.info("Synthesizing all findings")
        
        synthesis_prompt = f"""
        Create a comprehensive synthesis of all research findings. Organize the information into:
        
        1. **Key Research Areas Identified**
        2. **Recent Developments (Last 2 Years)**
        3. **Available Code Implementations**
        4. **Research Gaps and Opportunities**
        5. **Methodologies and Approaches**
        6. **Applications and Use Cases**
        
        Research Query: {state['query']}
        
        All Research Data:
        {json.dumps({
            "initial_results": state["initial_results"],
            "cross_references": state["cross_references"]
        }, indent=2)}
        
        Provide a detailed, well-structured synthesis that would be useful for a researcher or practitioner.
        """
        
        try:
            response = self.model.generate_content(synthesis_prompt)
            state["synthesis"] = response.text
            state["current_step"] = "synthesis_completed"
            
            logger.info("Synthesis completed")
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            state["synthesis"] = "Error occurred during synthesis"
        
        return state
    
    def _generate_follow_up(self, state: ResearchState) -> ResearchState:
        logger.info("Generating follow-up questions and recommendations")
        
        follow_up_prompt = f"""
        Based on the research synthesis, generate:
        
        1. **5-7 Follow-up Research Questions** that would advance the field
        2. **3-5 Practical Recommendations** for researchers or practitioners
        3. **2-3 Emerging Trends** to watch
        
        Research Synthesis:
        {state['synthesis']}
        
        Return the response in a structured format with clear sections.
        """
        
        try:
            response = self.model.generate_content(follow_up_prompt)
            state["follow_up_questions"] = [response.text]
            state["current_step"] = "follow_up_completed"
            
            logger.info("Follow-up questions generated")
            
        except Exception as e:
            logger.error(f"Error generating follow-up: {e}")
            state["follow_up_questions"] = ["Error occurred while generating follow-up questions"]
        
        return state
    
    def _final_synthesis(self, state: ResearchState) -> ResearchState:
        logger.info("Creating final research report")
        
        final_prompt = f"""
        Create a comprehensive final research report that includes:
        
        **Executive Summary**
        - Brief overview of findings
        
        **Research Methodology**
        - Sources consulted (arXiv, PapersWithCode, Google Scholar)
        - Search strategy and refinement process
        
        **Key Findings**
        - Main discoveries and insights
        
        **Detailed Analysis**
        {state['synthesis']}
        
        **Follow-up Research**
        {state['follow_up_questions'][0] if state['follow_up_questions'] else 'No follow-up questions generated'}
        
        **Recommendations**
        - For researchers
        - For practitioners
        - For future research directions
        
        **References**
        {state.get('formatted_references', 'No references available')}
        
        Original Query: {state['query']}
        Research Depth: {state['research_depth']} levels
        Total Sources Consulted: {len(state['initial_results'])} search iterations
        Total Unique Papers: {len(state.get('all_papers', []))}
        
        Format this as a professional research report.
        """
        
        try:
            response = self.model.generate_content(final_prompt)
            state["research_summary"] = response.text
            state["current_step"] = "final_synthesis_completed"
            
            logger.info("Final research report completed")
            
        except Exception as e:
            logger.error(f"Error in final synthesis: {e}")
            state["research_summary"] = "Error occurred while creating final report"
        
        return state
    
    def research(self, query: str, max_depth: int = 3) -> Dict[str, Any]:
        logger.info(f"Starting deep research for: {query}")
        initial_state = ResearchState(
            query=query,
            initial_results=[],
            refined_queries=[],
            cross_references=[],
            synthesis="",
            follow_up_questions=[],
            current_step="started",
            research_depth=0,
            max_depth=max_depth,
            all_findings=[],
            research_summary="",
            recommendations=[],
            all_papers=[],
            formatted_references=""
        )
        
        try:
            thread_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query) % 10000}"
            
            final_state = self.graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}}
            )
            logger.info("Deep research completed successfully")
            return final_state
        except Exception as e:
            logger.error(f"Error during research workflow: {e}")
            return {
                "error": str(e),
                "query": query,
                "research_summary": "Research failed due to an error"
            }

def main():
    assistant = DeepResearchAssistant()
    
    research_queries = [
        "diffusion models for time series forecasting",
        "large language models for code generation",
        "multimodal AI for medical diagnosis"
    ]
    
    query = "diffusion models for time series forecasting"
    print(f"\n{'='*60}")
    print(f"RESEARCH QUERY {query}")
    print(f"{'='*60}")
        
    try:
        results = assistant.research(query, max_depth=3)
            
        if "error" in results:
            print(f"Error: {results['error']}")
            
        print("\n--- FINAL RESEARCH REPORT ---")
        print(results.get("research_summary", "No report generated"))        
            
    except Exception as e:
        print(f"An error occurred during research: {e}")
        
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
