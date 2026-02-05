# Article-Atlas

This project transforms any news article into a knowledge graph by using web scraping, extraction, tokenization and LLMs. 

## Features

-  **Web Scraping** - Extract content from any article URL
-  **Smart Cleaning** - Remove ads, navigation, and promotional content
-  **Text Chunking** - Split articles into semantic chunks for processing
-  **AI Extraction** - Use GPT-4/Gemini to extract entities and relationships
-  **Graph Merging** - Deduplicate and merge entities across chunks
-  **Interactive Visualization** - Explore the knowledge graph with Cytoscape.js
-  **Export** - Download graphs as PNG or JSON

## Pipeline
<img width="425" height="485" alt="Screenshot 2026-02-04 at 01 48 58" src="https://github.com/user-attachments/assets/2492a82d-360f-4c16-a5ca-56e8646a1f9f" />

## Entity Types

The system extracts the following entity types:

- **PERSON** - People, historical figures, authors
- **ORGANIZATION** - Companies, institutions, government bodies
- **LOCATION** - Cities, countries, regions, places
- **CONCEPT** - Ideas, theories, topics, technologies
- **EVENT** - Historical events, occurrences, incidents
- **DATE** - Specific dates, time periods

## Relationship Types

Supported relationship types:

- **WORKS_AT** - Person works at Organization
- **LOCATED_IN** - Entity is located in Location
- **FOUNDED** - Person/Organization founded Entity
- **RELATED_TO** - General relationship between Concepts
- **PARTICIPATED_IN** - Person/Organization participated in Event
- **OCCURRED_ON** - Event occurred on Date
- **PART_OF** - Entity is part of another Entity

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Trafilatura** - Web scraping and content extraction
- **spaCy** - NLP and named entity recognition
- **LangChain** - Text splitting and chunking
- **OpenAI/Gemini** - LLM for entity extraction
- **Pydantic** - Data validation

### Frontend
- **React** - UI framework
- **Cytoscape.js** - Graph visualization
- **CSS3** - Styling and animations

## Demo Video
https://github.com/user-attachments/assets/1eacae89-c55a-4e21-a9f4-533cdbfcf364

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Trafilatura](https://trafilatura.readthedocs.io/) for web scraping
- [Cytoscape.js](https://js.cytoscape.org/) for graph visualization
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [OpenAI](https://openai.com/) for GPT models
- [Google](https://ai.google.dev/) for Gemini models

**Built with ❤️ for transforming articles into knowledge**
