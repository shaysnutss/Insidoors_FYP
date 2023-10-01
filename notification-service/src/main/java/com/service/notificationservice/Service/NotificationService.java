package com.service.notificationservice.Service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class NotificationService {

    // This method will send a GET request to account service to get an account object
    public Map<String,Object> getAccount(String apiURL){
        HttpClient client = HttpClients.createDefault();
        HttpGet request = new HttpGet(apiURL);
        Map<String,Object> map = null;
        try{
            HttpResponse response = client.execute(request);
            if(response.getStatusLine().getStatusCode() != 200){
                throw new RuntimeException("Failed : HTTP error code : "
                        + response.getStatusLine().getStatusCode());
            }
            HttpEntity entity = response.getEntity();
            String responseString = EntityUtils.toString(entity, "UTF-8");
            ObjectMapper mapper = new ObjectMapper();
            map = mapper.readValue(responseString, new TypeReference<Map<String,Object>>(){});


        }catch(Exception e){
            e.printStackTrace();
            return null;
        }
        return map;

    }
}
