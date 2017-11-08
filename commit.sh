echo "Enter commit message:"
read message 
git add --all && git commit -m "$message"
git push
