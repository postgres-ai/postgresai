{{/*
Expand the name of the chart.
*/}}
{{- define "postgres-ai-monitoring.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
Truncate to 45 chars to leave room for component suffixes (e.g. -victoriametrics)
*/}}
{{- define "postgres-ai-monitoring.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 45 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 45 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 45 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "postgres-ai-monitoring.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "postgres-ai-monitoring.labels" -}}
helm.sh/chart: {{ include "postgres-ai-monitoring.chart" . }}
{{ include "postgres-ai-monitoring.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Build connection string from database configuration
Password is retrieved from secrets using passwordSecretKey
*/}}
{{- define "postgres-ai-monitoring.dbConnStr" -}}
{{- $db := .db -}}
{{- $root := .root -}}
{{- if $db.connStr }}
{{- $db.connStr }}
{{- else }}
{{- $host := $db.host | default "localhost" }}
{{- $port := $db.port | default 5432 }}
{{- $database := $db.database | default "postgres" }}
{{- $user := $db.user | default "postgres" }}
{{- $passwordKey := printf "db-password-%s" $db.passwordSecretKey }}
postgresql://{{ $user }}:$(DB_PASSWORD)@{{ $host }}:{{ $port }}/{{ $database }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "postgres-ai-monitoring.selectorLabels" -}}
app.kubernetes.io/name: {{ include "postgres-ai-monitoring.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "postgres-ai-monitoring.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "postgres-ai-monitoring.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Namespace
*/}}
{{- define "postgres-ai-monitoring.namespace" -}}
{{- default .Release.Namespace .Values.namespaceOverride }}
{{- end }}

{{/*
Determine effective cluster name with fallbacks.
*/}}
{{- define "postgres-ai-monitoring.effectiveClusterName" -}}
{{- if .Values.reporter.clusterName }}
{{- .Values.reporter.clusterName }}
{{- else if .Values.global.clusterName }}
{{- .Values.global.clusterName }}
{{- else }}
k8s-cluster
{{- end }}
{{- end }}

{{/*
Determine effective node name with fallbacks.
*/}}
{{- define "postgres-ai-monitoring.effectiveNodeName" -}}
{{- if .Values.reporter.nodeName }}
{{- .Values.reporter.nodeName }}
{{- else if .Values.global.nodeName }}
{{- .Values.global.nodeName }}
{{- else }}
{{- "" }}
{{- end }}
{{- end }}

{{/*
Get cluster name for a specific database with fallbacks.
*/}}
{{- define "postgres-ai-monitoring.databaseClusterName" -}}
{{- $db := .db -}}
{{- $root := .root -}}
{{- if $db.clusterName }}
{{- $db.clusterName }}
{{- else if $root.Values.reporter.clusterName }}
{{- $root.Values.reporter.clusterName }}
{{- else if $root.Values.global.clusterName }}
{{- $root.Values.global.clusterName }}
{{- else }}
{{- "k8s-cluster" }}
{{- end }}
{{- end }}

{{/*
Get node name for a specific database with fallbacks.
*/}}
{{- define "postgres-ai-monitoring.databaseNodeName" -}}
{{- $db := .db -}}
{{- $root := .root -}}
{{- if $db.nodeName }}
{{- $db.nodeName }}
{{- else if $root.Values.reporter.nodeName }}
{{- $root.Values.reporter.nodeName }}
{{- else if $root.Values.global.nodeName }}
{{- $root.Values.global.nodeName }}
{{- else }}
{{- "" }}
{{- end }}
{{- end }}

{{/*
Get unique cluster/node combinations from monitoredDatabases.
Returns a list of dicts with cluster and nodeName keys.
*/}}
{{- define "postgres-ai-monitoring.uniqueClusterNodeCombinations" -}}
{{- $root := . -}}
{{- $combinations := list -}}
{{- $seen := dict -}}
{{- range $db := .Values.monitoredDatabases }}
  {{- $clusterName := include "postgres-ai-monitoring.databaseClusterName" (dict "db" $db "root" $root) | trim -}}
  {{- $nodeName := include "postgres-ai-monitoring.databaseNodeName" (dict "db" $db "root" $root) | trim -}}
  {{- $key := printf "%s|%s" $clusterName $nodeName -}}
  {{- if not (hasKey $seen $key) }}
    {{- $_ := set $seen $key true -}}
    {{- $combinations = append $combinations (dict "cluster" $clusterName "nodeName" $nodeName) -}}
  {{- end }}
{{- end }}
{{- if eq (len $combinations) 0 }}
  {{- $clusterName := include "postgres-ai-monitoring.effectiveClusterName" $root | trim -}}
  {{- $nodeName := include "postgres-ai-monitoring.effectiveNodeName" $root | trim -}}
  {{- $combinations = append $combinations (dict "cluster" $clusterName "nodeName" $nodeName) -}}
{{- end }}
{{- $combinations | toJson }}
{{- end }}

{{/*
Get the secret name to use.
Returns existingSecret.name if set, otherwise returns the default secret name.
*/}}
{{- define "postgres-ai-monitoring.secretName" -}}
{{- $existingSecretName := "" }}
{{- if .Values.existingSecret }}
  {{- $existingSecretName = .Values.existingSecret.name | default "" }}
{{- end }}
{{- if and $existingSecretName (ne $existingSecretName "") }}
{{- $existingSecretName }}
{{- else }}
{{- printf "%s-secrets" (include "postgres-ai-monitoring.fullname" .) }}
{{- end }}
{{- end }}


