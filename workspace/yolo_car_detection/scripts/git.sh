cd /root/.ssh
ssh-keygen -t rsa -C "개인정보@gmail.com" -f "id_rsa_cuffyluv"
eval "$(ssh-agent -s)" 
ssh-add id_rsa_cuffyluv
cat id_rsa_cuffyluv.pub
code config
Host github.com-cuffyluv
  HostName github.com
  User git
  IdentityFile /root/.ssh/id_rsa_cuffyluv
ssh -T git@github.com-cuffyluv
git clone git@github.com-cuffyluv:CV-11-boostcamp-ai-tech-8th/black-duck-model.git
git remote set-url origin git@github.com-cuffyluv:CV-11-boostcamp-ai-tech-8th/black-duck-model.git
git remote -v
git config user.name "cuffyluv"
git config user.email "90530238+cuffyluv@users.noreply.github.com"