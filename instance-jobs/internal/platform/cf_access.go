package platform

import (
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
)

func accessOrigin(raw string) string {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "https" || u.Hostname() == "" {
		return ""
	}
	port := 443
	if u.Port() != "" {
		port, err = strconv.Atoi(u.Port())
		if err != nil || port < 0 || port > 65535 {
			return ""
		}
	}
	return u.Scheme + "://" + net.JoinHostPort(strings.ToLower(u.Hostname()), strconv.Itoa(port))
}

func (c *Client) addAccessHeaders(req *http.Request) {
	id := strings.TrimSpace(os.Getenv("CF_ACCESS_CLIENT_ID"))
	secret := strings.TrimSpace(os.Getenv("CF_ACCESS_CLIENT_SECRET"))
	if id == "" || secret == "" {
		return
	}
	origin := accessOrigin(req.URL.String())
	if origin == "" || origin != accessOrigin(c.baseURL) {
		return
	}
	req.Header.Set("CF-Access-Client-Id", id)
	req.Header.Set("CF-Access-Client-Secret", secret)
}
