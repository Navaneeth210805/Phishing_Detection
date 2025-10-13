# 🔑 API Keys Setup Guide

This guide will help you configure API keys to unlock the full potential of the crawler system.

## 📋 Required APIs

### 1. Google Custom Search API (Recommended)
**Benefits**: Find suspicious websites through Google search
**Free Tier**: 100 queries/day

#### Setup Steps:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable "Custom Search API"
4. Go to "Credentials" → "Create Credentials" → "API Key"
5. Copy your API key

#### Create Custom Search Engine:
1. Go to [Google Custom Search](https://cse.google.com/)
2. Click "Add" to create new search engine
3. In "Sites to search": Enter `*` (search entire web)
4. Click "Create"
5. Go to "Setup" → "Basics" → Copy "Search engine ID"

### 2. PhishTank API (Optional)
**Benefits**: Higher rate limits for phishing database
**Free Tier**: Enhanced access to verified phishing URLs

#### Setup Steps:
1. Register at [PhishTank.com](https://www.phishtank.com/)
2. Go to "Developer Info" → "Manage Applications"
3. Create new application
4. Copy your API key

### 3. URLScan.io API (Optional)
**Benefits**: Submit URLs for scanning, access private scans
**Free Tier**: 1000 scans/day

#### Setup Steps:
1. Register at [URLScan.io](https://urlscan.io/)
2. Go to "Settings" → "API"
3. Generate API key
4. Copy your API key

## 🛠️ Configuration

### Step 1: Create .env file
```bash
cd /home/vk/Phishing_Detection/crawlers
cp .env.example .env
```

### Step 2: Add your API keys to .env
```env
# Google Custom Search API
GOOGLE_API_KEY=AIzaSyD...your_actual_api_key_here
GOOGLE_SEARCH_ENGINE_ID=017576662...your_search_engine_id

# PhishTank API (Optional)
PHISHTANK_API_KEY=your_phishtank_api_key_here

# URLScan.io API (Optional)
URLSCAN_API_KEY=your_urlscan_api_key_here
```

### Step 3: Check configuration
```bash
python main.py api-status
```

## 🎯 Priority Levels

### Without API Keys (Limited Mode)
- ✅ CT Logs Crawler (Works fully)
- ✅ Typosquatting Crawler (Works fully)
- ❌ Google Search Crawler (Demo mode only)
- ⚠️ PhishTank Crawler (Limited rate)
- ⚠️ URLScan.io Crawler (Public scans only)

### With API Keys (Full Mode)
- ✅ All crawlers fully functional
- ✅ Higher rate limits
- ✅ Better data quality
- ✅ More comprehensive coverage

## 🚀 Testing Your Setup

### Test API Configuration
```bash
python main.py api-status
```

### Run Single Crawl
```bash
python main.py single
```

### Start Continuous Crawling
```bash
python main.py run
```

## 🔒 Security Notes

1. **Never commit .env files** to version control
2. **Use environment variables** in production
3. **Rotate API keys** regularly
4. **Monitor API usage** to avoid rate limits

## 💡 Cost Considerations

All mentioned APIs have generous free tiers:
- **Google Custom Search**: 100 queries/day free
- **PhishTank**: Free for research/educational use
- **URLScan.io**: 1000 scans/day free

For production use, consider upgrading to paid tiers for higher limits.

## 🆘 Troubleshooting

### "API key not configured" warnings
- Check your .env file exists and has correct format
- Verify API keys are valid and active
- Restart crawler system after changes

### Rate limit errors
- Reduce crawler frequencies
- Upgrade to paid API tiers
- Use multiple API keys (advanced)

### No results from Google Search
- Verify Custom Search Engine is configured
- Check search engine ID is correct
- Ensure API has Custom Search enabled
