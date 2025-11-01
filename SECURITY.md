# 🔐 Security Setup

## ⚠️ IMPORTANT: Environment Variables Setup

This application requires sensitive API keys that should **NEVER** be committed to version control.

### Setup Steps:

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Get your Google Gemini API Key:**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

3. **Update `.env` file:**
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **Verify `.env` is in `.gitignore`:**
   - Check that `.env` is listed in `.gitignore`
   - This prevents accidentally committing secrets

### ⚠️ Security Best Practices:

- ✅ **DO:** Keep `.env` file local only
- ✅ **DO:** Share `.env.example` without real values
- ✅ **DO:** Use environment variables for all secrets
- ❌ **DON'T:** Commit `.env` to git
- ❌ **DON'T:** Share API keys in code or chat
- ❌ **DON'T:** Use production keys in development

### If You Accidentally Commit a Secret:

1. **Immediately revoke the exposed key** in Google Cloud Console
2. **Generate a new key**
3. **Remove from git history** using:
   ```bash
   # Use BFG Repo-Cleaner or git filter-branch
   # Or create a new repository with clean history
   ```
4. **Update your `.env` with the new key**

---

**Remember:** Security is everyone's responsibility! 🔒
