'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { 
  apiClient, 
  type SystemStatus, 
  type CSE, 
  type DomainClassification, 
  type RecentDetection,
  handleApiError,
  isApiResponseSuccess 
} from '@/lib/api';

export default function Dashboard() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [cses, setCSEs] = useState<Record<string, CSE>>({});
  const [recentDetections, setRecentDetections] = useState<RecentDetection[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Domain classification form
  const [domainInput, setDomainInput] = useState('');
  const [selectedCSE, setSelectedCSE] = useState('');
  const [classificationResult, setClassificationResult] = useState<DomainClassification | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);

  // Clean domain input as user types
  const handleDomainInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    let value = e.target.value;
    
    // Real-time cleaning
    value = value.replace(/^https?:\/\//, ''); // Remove protocols
    value = value.replace(/\/.*$/, ''); // Remove paths
    value = value.replace(/^www\./, ''); // Remove www prefix
    
    setDomainInput(value);
  };

  // Load initial data
  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Load system status
      const statusResponse = await apiClient.getSystemStatus();
      if (isApiResponseSuccess(statusResponse)) {
        setSystemStatus(statusResponse.data);
      } else {
        throw new Error(statusResponse.error || 'Failed to load system status');
      }

      // Load CSEs
      const cseResponse = await apiClient.getAllCSEs();
      if (isApiResponseSuccess(cseResponse)) {
        setCSEs(cseResponse.data);
      } else {
        throw new Error(cseResponse.error || 'Failed to load CSEs');
      }

      // Load dashboard stats for recent detections
      const statsResponse = await apiClient.getDashboardStats();
      if (isApiResponseSuccess(statsResponse)) {
        setRecentDetections(statsResponse.data.recent_detections);
      }

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      handleApiError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDomainClassification = async () => {
    if (!domainInput.trim()) return;
    
    setIsClassifying(true);
    setClassificationResult(null);

    try {
      // Clean the domain input - remove protocol and trailing slashes
      let cleanDomain = domainInput.trim();
      cleanDomain = cleanDomain.replace(/^https?:\/\//, ''); // Remove http:// or https://
      cleanDomain = cleanDomain.replace(/\/.*$/, ''); // Remove path and everything after
      cleanDomain = cleanDomain.replace(/^www\./, ''); // Remove www. prefix
      
      // Update the input field with the clean domain
      setDomainInput(cleanDomain);
      
      const response = await apiClient.classifyDomain(
        cleanDomain,
        selectedCSE || undefined
      );
      
      if (isApiResponseSuccess(response)) {
        setClassificationResult(response.data);
      } else {
        throw new Error(response.error || 'Classification failed');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Classification error';
      setError(errorMessage);
      handleApiError(errorMessage);
    } finally {
      setIsClassifying(false);
    }
  };

  const toggleMonitoring = async () => {
    if (!systemStatus) return;

    try {
      const response = systemStatus.monitoring_active
        ? await apiClient.stopMonitoring()
        : await apiClient.startMonitoring();

      if (isApiResponseSuccess(response)) {
        // Reload system status
        const statusResponse = await apiClient.getSystemStatus();
        if (isApiResponseSuccess(statusResponse)) {
          setSystemStatus(statusResponse.data);
        }
      } else {
        throw new Error(response.error || 'Failed to toggle monitoring');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Monitoring toggle failed';
      setError(errorMessage);
      handleApiError(errorMessage);
    }
  };

  const getClassificationBadgeVariant = (classification: string) => {
    switch (classification) {
      case 'Phishing':
        return 'destructive' as const;
      case 'Suspected':
        return 'secondary' as const;
      case 'Legitimate':
        return 'default' as const;
      default:
        return 'outline' as const;
    }
  };

  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-lg">Loading dashboard...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Phishing Detection Dashboard</h1>
        <Button onClick={loadDashboardData} variant="outline" className="cursor-pointer">
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* System Overview */}
      {systemStatus && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Total CSEs</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStatus.total_cses}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Detected Domains</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStatus.detected_domains}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Monitoring Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-2">
                <Badge variant={systemStatus.monitoring_active ? 'default' : 'secondary'}>
                  {systemStatus.monitoring_active ? 'Active' : 'Inactive'}
                </Badge>
                <Button
                  onClick={toggleMonitoring}
                  size="sm"
                  variant="outline"
                  className="cursor-pointer"
                >
                  {systemStatus.monitoring_active ? 'Stop' : 'Start'}
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Model Status</CardTitle>
            </CardHeader>
            <CardContent>
              <Badge variant={systemStatus.model_loaded ? 'default' : 'destructive'}>
                {systemStatus.model_loaded ? 'Loaded' : 'Not Loaded'}
              </Badge>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="classify" className="space-y-4">
        <TabsList className="cursor-pointer">
          <TabsTrigger value="classify" className="cursor-pointer">Domain Classification</TabsTrigger>
          <TabsTrigger value="cses" className="cursor-pointer">CSE Management</TabsTrigger>
          <TabsTrigger value="detections" className="cursor-pointer">Recent Detections</TabsTrigger>
        </TabsList>

        <TabsContent value="classify" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Classify Domain</CardTitle>
              <CardDescription>
                Enter a domain to check if it&apos;s potentially phishing a CSE
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex space-x-4">
                <div className="flex-1">
                  <Label htmlFor="domain">Domain</Label>
                  <Input
                    id="domain"
                    placeholder="sbi.co.in (domain only, no http/https)"
                    value={domainInput}
                    onChange={handleDomainInputChange}
                    className="cursor-text"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Enter domain without protocol (http/https) - e.g., &quot;sbi.co.in&quot; not &quot;https://sbi.co.in&quot;
                  </p>
                </div>
                <div className="w-64">
                  <Label htmlFor="cse">Target CSE (Optional)</Label>
                  <select
                    id="cse"
                    className="w-full h-10 px-3 py-2 border border-input bg-background rounded-md cursor-pointer"
                    value={selectedCSE}
                    onChange={(e) => setSelectedCSE(e.target.value)}
                  >
                    <option value="">Auto-detect</option>
                    {Object.keys(cses).map((cseName) => (
                      <option key={cseName} value={cseName}>
                        {cseName}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex items-end">
                  <Button
                    onClick={handleDomainClassification}
                    disabled={isClassifying || !domainInput.trim()}
                    className="cursor-pointer disabled:cursor-not-allowed"
                  >
                    {isClassifying ? 'Classifying...' : 'Classify'}
                  </Button>
                </div>
              </div>

              {classificationResult && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <span>Classification Result</span>
                      <Badge variant={getClassificationBadgeVariant(classificationResult.classification.classification)}>
                        {classificationResult.classification.classification}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Confidence</Label>
                        <div className="text-lg font-semibold">
                          {(classificationResult.classification.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <Label>Risk Score</Label>
                        <div className="text-lg font-semibold">
                          {classificationResult.classification.risk_score.toFixed(2)}
                        </div>
                      </div>
                    </div>
                    
                    {classificationResult.target_cse && (
                      <div>
                        <Label>Target CSE</Label>
                        <div className="font-medium">{classificationResult.target_cse}</div>
                      </div>
                    )}

                    <div>
                      <Label>Reasoning</Label>
                      <ul className="list-disc list-inside space-y-1">
                        {classificationResult.classification.reasoning.map((reason, index) => (
                          <li key={index} className="text-sm">{reason}</li>
                        ))}
                      </ul>
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="cses" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Critical Sector Entities (CSEs)</CardTitle>
              <CardDescription>
                Manage the organizations being monitored for phishing attempts
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Sector</TableHead>
                    <TableHead>Domains</TableHead>
                    <TableHead>Keywords</TableHead>
                    <TableHead>Description</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(cses).map(([name, cse]) => (
                    <TableRow key={name}>
                      <TableCell className="font-medium">{name}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{cse.sector}</Badge>
                      </TableCell>
                      <TableCell>{cse.domain_count}</TableCell>
                      <TableCell>{cse.keyword_count}</TableCell>
                      <TableCell className="max-w-xs truncate">
                        {cse.description}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="detections" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Detections</CardTitle>
              <CardDescription>
                Latest phishing attempts detected by the system
              </CardDescription>
            </CardHeader>
            <CardContent>
              {recentDetections.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Domain</TableHead>
                      <TableHead>Classification</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Target CSE</TableHead>
                      <TableHead>Risk Score</TableHead>
                      <TableHead>Timestamp</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {recentDetections.map((detection, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">{detection.domain}</TableCell>
                        <TableCell>
                          <Badge variant={getClassificationBadgeVariant(detection.classification)}>
                            {detection.classification}
                          </Badge>
                        </TableCell>
                        <TableCell>{(detection.confidence * 100).toFixed(1)}%</TableCell>
                        <TableCell>{detection.target_cse}</TableCell>
                        <TableCell>{detection.risk_score.toFixed(2)}</TableCell>
                        <TableCell>
                          {new Date(detection.timestamp).toLocaleString()}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No recent detections found
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
