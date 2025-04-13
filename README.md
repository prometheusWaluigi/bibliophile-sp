# BibliophileSP

SP-API powered inventory sync and analysis tool for book sellers.

## 📦 Project Overview

BibliophileSP is a secure, performant, read-only pipeline to ingest Amazon SP-API data (sales + inventory), analyze trends, and flag stale or low-quality book listings using local compute acceleration (oneAPI).

## 🧠 Key Features

- **SP-API Sales Data Pull**: Use Reports API to retrieve sales history
- **Inventory Snapshot Sync**: Use Listings + Catalog API to get current book metadata & availability
- **Trend Detection Engine**: Use pandas + NumPy to identify stale inventory and sales velocity patterns
- **Quality Flagging Logic**: Heuristics to flag bad listings (e.g., missing ISBN, poor titles, no sales)
- **Local Acceleration Layer**: Optional oneAPI-accelerated modules for NumPy-heavy operations
- **Sandbox-first Dev Mode**: Use SP-API sandbox to build & test request flows before touching live data

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- Poetry (for dependency management)
- Docker (optional, for containerized deployment)

### Installation

#### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bibliophile-sp.git
   cd bibliophile-sp
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   ```bash
   cp .env.template .env
   ```
   Then edit the `.env` file with your Amazon SP-API credentials.

4. (Optional) Install oneAPI acceleration libraries:
   ```bash
   pip install daal4py scikit-learn-intelex
   ```

#### Option 2: Docker Installation (Recommended)

This option automatically sets up all dependencies including oneAPI acceleration:

1. Build and run using Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. Or manually with Docker:
   ```bash
   docker build -t bibliophile-sp .
   docker run -it -v $(pwd)/output:/app/output --env-file .env bibliophile-sp
   ```

### Running the Application

#### Using Poetry (Local Installation)

Run with basic functionality:
```bash
poetry run python src/main.py
```

Run with SP-API integration:
```bash
poetry run python src/main.py --use-spapi
```

Run with CLI interface and more options:
```bash
poetry run python src/cli.py --help
```

#### Using Docker

```bash
# Basic run
docker-compose up

# With custom options
docker-compose run --rm bibiliophile-sp poetry run python src/cli.py --use-spapi
```

## ⚡ Performance Acceleration

BibliophileSP intelligently uses hardware acceleration on supported platforms:

### Intel oneAPI Acceleration

On Intel processors with oneAPI libraries installed:

- **Automatic Detection**: The application automatically detects and uses oneAPI libraries if available
- **Accelerated Algorithms**: K-means clustering and other algorithms can be up to 10x faster
- **Docker Support**: The included Dockerfile automatically sets up oneAPI acceleration

To verify oneAPI is active, look for the "⚡ Intel oneAPI acceleration is available" message when running the application.

### Apple Silicon / macOS Acceleration

On macOS systems:

- **Apple Accelerate Framework**: Automatically uses Apple's high-performance Accelerate framework on Apple Silicon (M1/M2/M3) and Intel Macs
- **Native Performance**: Takes advantage of Apple's optimized BLAS implementation for linear algebra operations
- **Docker Compatibility**: Use `Dockerfile.mac` for macOS-friendly containerization (note: for full Accelerate framework acceleration, run natively on macOS)

To verify Apple Accelerate is active, look for the "⚡ Apple Accelerate framework is available" message when running the application.

### Cross-Platform Support

The application will automatically detect the available acceleration features and use the most appropriate one for your platform. You can also:

- Disable specific acceleration with `--no-oneapi` or `--no-apple-accelerate`
- Disable all acceleration with `--no-acceleration`
- Get installation instructions for Apple Accelerate with `--install-numpy-accelerate`

## 📊 Output

The application generates a CSV file with inventory analysis, including:

| SKU | Title | Sales Last 30d | Days Since Last Sale | Flag | Notes |
|-----|-------|----------------|-----------------------|------|-------|
| B000FJS1B4 | *The Hobbit* | 0 | 189 | ⚠️ | Reprice or remove |
| 0385472579 | *Zen Mind* | 14 | 3 | ✅ | Good seller |

## 🧬 Analysis Logic

1. **Stale Detection**:  
   - `days_since_last_sale > 120`  
   - `total_sales_last_12mo < 3`

2. **Bad Metadata**:  
   - Title length < 5 chars  
   - No ISBN  
   - No cover image  
   - Price < $2

3. **Heuristic Score (0–1)**:  
   - Composite metric for easy sorting in dashboards

## 🔐 Security / Privacy

- No write permissions required
- Local-only storage of credentials
- OneAPI runs on-device, no external compute
