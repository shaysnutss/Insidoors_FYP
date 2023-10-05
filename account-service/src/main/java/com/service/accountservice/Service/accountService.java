package com.service.accountservice.Service;

import org.apache.coyote.Response;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPut;
import org.apache.http.client.utils.URIBuilder;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestBody;

import java.net.URI;

@Service
public class accountService {

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

    public ResponseEntity<?> putMethod(String apiUrl,@RequestBody String requestBody ){
        try {
            URI uri = new URIBuilder(apiUrl)
                    .setParameter("data", requestBody)
                    .build();

            HttpClient client = HttpClients.createDefault();
            HttpPut request = new HttpPut(apiUrl);
            StringEntity entity = new StringEntity(requestBody);
            request.setEntity(entity);
            HttpResponse response = client.execute(request);

            String responseBody = EntityUtils.toString(response.getEntity());

            return ResponseEntity.ok(responseBody);

        } catch (Exception e) {
            return ResponseEntity.status(500).build();
        }
    }


}
