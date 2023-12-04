package com.service.schedulerservice.scheduler;

import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.concurrent.TimeUnit;


@Component
public class schedulingAPI {

    @Value("${PC_ACCESS_API_TRAIN_URL}")
    private String pcAccessApiTrainUrl;

    @Value("${BUILDING_ACCESS_API_TRAIN_URL}")
    private String buildingAccessApiTrainUrl;

    @Value("${PROXY_API_TRAIN_URL}")
    private String proxyApiTrainUrl;

    @Value("${PC_ACCESS_API_BASE_URL}")
    private String pcAccessApiBaseUrl;

    @Value("${BUILDING_ACCESS_API_BASE_URL}")
    private String buildingAccessApiBaseUrl;

    @Value("${PROXY_API_BASE_URL}")
    private String proxyApiBaseUrl;

    @Value("${RBA_BA_API_BASE_URL}")
    private String postToBAApiBaseUrl;

    @Scheduled(cron = "0 0 1 * *") // At 12:00 AM, on day 1 of the month
    public void trainAPIs(){
        // Print the current time and date
        System.out.println("The scheduler service API is running at " + java.time.LocalDateTime.now());
        // Call the APIs

        // First API called is pc access
        ResponseEntity responseEntity = getMethod(pcAccessApiTrainUrl);
        String bodyAPIcall = responseEntity.getBody().toString();
        System.out.println("The response from the pc access API is " + bodyAPIcall);

        // Second API called is building access
        responseEntity = getMethod(buildingAccessApiTrainUrl);
        bodyAPIcall = responseEntity.getBody().toString();
        System.out.println("The response from the building access API is " + bodyAPIcall);

        // Third API called is proxy logs
        responseEntity = getMethod(proxyApiTrainUrl);
        bodyAPIcall = responseEntity.getBody().toString();
        System.out.println("The response from the proxy API is " + bodyAPIcall);
    }
    
    
    
    @Scheduled(fixedRate = 10, initialDelay = 10, timeUnit = TimeUnit.MINUTES)
    public void schedulingAPIs(){
        // Print the current time and date
        System.out.println("The scheduler service API is running at " + java.time.LocalDateTime.now());
        // Call the APIs

        // First API called is pc access
        ResponseEntity responseEntity = getMethod(pcAccessApiBaseUrl);
        String bodyAPIcall = responseEntity.getBody().toString();
        System.out.println("The response from the pc access API is " + bodyAPIcall);

        // Second API called is building access
        responseEntity = getMethod(buildingAccessApiBaseUrl);
        bodyAPIcall = responseEntity.getBody().toString();
        System.out.println("The response from the building access API is " + bodyAPIcall);

        // Third API called is proxy logs
        responseEntity = getMethod(proxyApiTrainUrl);
        bodyAPIcall = responseEntity.getBody().toString();
        System.out.println("The response from the proxy API is " + bodyAPIcall);

        responseEntity = getMethod(postToBAApiBaseUrl);
        bodyAPIcall = responseEntity.getBody().toString();
        System.out.println("The response from the posting to behavioral analysis is " + bodyAPIcall);

    }

    public ResponseEntity<?> getMethod(String apiUrl){
        try {
            HttpClient client = HttpClients.createDefault();
            HttpGet request = new HttpGet(apiUrl);
            HttpResponse response = client.execute(request);

            String responseBody = EntityUtils.toString(response.getEntity());

            return ResponseEntity.ok(responseBody);

        } catch (Exception e) {
            return ResponseEntity.status(500).build();
        }
    }
}
