#!/bin/bash
set -e

rm -rf audit_workspace
mkdir -p audit_workspace
cd audit_workspace

repos=(
  "package-integrity-classification-via-sim-to-real"
  "agentic-systems-verifier"
  "camirian"
  "safe-acs"
  "citadel-os"
  "articulated-robot-manipulation"
  "sim-to-real-control-systems"
  "distributed-robotics-infrastructure"
  "career-vault"
  "coursera-ml-specialization"
  "robotics-ontology"
)

echo "# Deep Dive Repository Audit Report" > audit_report.md

for repo in "${repos[@]}"; do
  echo "========================================"
  echo "Auditing $repo"
  echo "========================================"
  echo "## [$repo](https://github.com/camirian/$repo)" >> audit_report.md
  
  if ! gh repo clone "camirian/$repo" "$repo"; then
    echo "Failed to clone $repo - skipping"
    echo "Failed to clone" >> audit_report.md
    continue
  fi
  
  cd "$repo"
  
  echo "### 1. Active Redundant/Unneeded Folders" >> ../audit_report.md
  find . -type d \( -name "*temp*" -o -name "*tmp*" -o -name "__pycache__" -o -name "isaac_ros_common" -o -name "node_modules" \) -not -path "*/.git/*" > redundant.txt || true
  if [ -s redundant.txt ]; then
    sed 's/^/- /' redundant.txt >> ../audit_report.md
  else
    echo "*None found*" >> ../audit_report.md
  fi
  
  echo "### 2. Active Potential Secrets / Data Spills" >> ../audit_report.md
  find . -type f \( -name ".env*" -o -name "*.pem" -o -name "*.key" -o -name "credentials*.json" -o -name "secrets*.yaml" \) -not -path "*/.git/*" > secrets_files.txt || true
  if [ -s secrets_files.txt ]; then
    sed 's/^/- /' secrets_files.txt >> ../audit_report.md
  else
    echo "*No suspicious files found*" >> ../audit_report.md
  fi

  echo "### 3. Historical Data Spills (in git log)" >> ../audit_report.md
  git log --all --name-status --pretty=format: | grep -E -i "\.env|\.pem|\.key|credential.*\.json|secret|__pycache__|temp|tmp|isaac_ros_common" | awk '{print $2}' | sort -u > history_spills.txt || true
  if [ -s history_spills.txt ]; then
    echo "\`\`\`text" >> ../audit_report.md
    head -n 20 history_spills.txt >> ../audit_report.md
    if [ $(wc -l < history_spills.txt) -gt 20 ]; then echo "...(truncated)" >> ../audit_report.md; fi
    echo "\`\`\`" >> ../audit_report.md
  else
    echo "*Clean history*" >> ../audit_report.md
  fi
  
  echo "### 4. Code Grep for Hardcoded Tokens" >> ../audit_report.md
  grep -rE "sk-ant-|sk-proj-|ghp_|AKIA|passwd|password|api_key|secret_key|x-api-key" . --exclude-dir=.git > tokens.txt || true
  if [ -s tokens.txt ]; then
    echo "Found potential hardcoded tokens or passwords in these files:" >> ../audit_report.md
    awk -F: '{print "- " $1}' tokens.txt | sort -u >> ../audit_report.md
  else
    echo "*No hardcoded tokens detected*" >> ../audit_report.md
  fi
  
  cd ..
done

echo "Audit completed successfully."
