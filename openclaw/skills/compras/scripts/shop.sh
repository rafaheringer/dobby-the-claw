#!/bin/bash
# Shopping helper: types secrets from 1Password straight into the browser, so they never
# show up in the conversation, and reads verification codes from the Dobby Gmail.
set -euo pipefail
export PATH=/home/dobby/.nvm/versions/node/v24.22.0/bin:$PATH
set -a; . /home/dobby/.config/op/dobby.env; set +a
VAULT="Dobby Vault"
DIR=$(cd "$(dirname "$0")" && pwd)

usage() {
  cat <<'U'
shop.sh user <Item>                                   username of a login (Amazon, "Mercado Livre")
shop.sh type <Item> <field> <ref> [--target-id <tab>] type a secret into a browser field
shop.sh select <Item> <field> <ref> [--target-id <tab>] choose a secret in a <select>
shop.sh emails "<gmail query>"                        list recent e-mails (from | subject | preview)
shop.sh email "<gmail query>"                         full text of the newest matching e-mail
fields: password | otp | ccnum | cardholder | cvv | expiry-mmyy | expiry-mm | expiry-yy | expiry-yyyy
U
  exit 1
}

secret() {  # <Item> <field>
  local item=$1 field=$2 e
  case $field in
    password) op item get "$item" --vault "$VAULT" --fields password --reveal </dev/null ;;
    otp) op item get "$item" --vault "$VAULT" --otp </dev/null ;;
    ccnum) op item get "Credit Card" --vault "$VAULT" --fields ccnum --reveal </dev/null | tr -d ' ' ;;
    cardholder|cvv) op item get "Credit Card" --vault "$VAULT" --fields "$field" --reveal </dev/null ;;
    expiry-*)
      e=$(op item get "Credit Card" --vault "$VAULT" --fields expiry --reveal </dev/null | tr -dc 0-9)
      local m y
      if [ "${e:0:2}" = "20" ]; then y=${e:0:4}; m=${e:4:2}; else m=${e:0:2}; y=${e:2:4}; fi
      case $field in
        expiry-mmyy) echo "$m${y:2:2}" ;; expiry-mm) echo "$m" ;;
        expiry-yy) echo "${y:2:2}" ;; expiry-yyyy) echo "$y" ;; *) usage ;;
      esac ;;
    *) usage ;;
  esac
}

cmd=${1:-}; shift || true
case $cmd in
  user) op item get "$1" --vault "$VAULT" --fields username </dev/null ;;
  type|select)
    [ $# -ge 3 ] || usage
    item=$1 field=$2 ref=$3; shift 3
    v=$(secret "$item" "$field")
    openclaw browser --timeout 120000 --browser-profile compras "$cmd" "$@" "$ref" "$v" >/dev/null
    echo "ok: $field typed into $ref" ;;
  emails) python3 "$DIR/gmail_latest.py" "${1:-newer_than:15m}" ;;
  email) python3 "$DIR/gmail_body.py" "${1:-newer_than:15m}" ;;
  *) usage ;;
esac
