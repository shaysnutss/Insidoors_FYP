// package com.service.accountservice;

// import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
// import com.fasterxml.jackson.annotation.PropertyAccessor;
// import com.fasterxml.jackson.annotation.JsonAutoDetect.Visibility;
// import com.fasterxml.jackson.databind.DeserializationFeature;
// import com.fasterxml.jackson.databind.ObjectMapper;

// import org.springframework.boot.test.context.TestConfiguration;
// import org.springframework.context.annotation.Bean;
// import org.springframework.security.core.GrantedAuthority;
// import org.springframework.test.web.servlet.MockMvc;
// import org.springframework.test.web.servlet.setup.MockMvcBuilders;
// import org.springframework.web.context.WebApplicationContext;
// import com.fasterxml.jackson.databind.module.SimpleModule;


// @TestConfiguration
// @JsonIgnoreProperties(ignoreUnknown = true)
// public class TestConfig {

//     @Bean
//     public MockMvc mockMvc(WebApplicationContext webApplicationContext) {
//         return MockMvcBuilders.webAppContextSetup(webApplicationContext).build();
//     }

//     @Bean
//     public ObjectMapper objectMapper() {
//         ObjectMapper objectMapper = new ObjectMapper();
//         SimpleModule module = new SimpleModule();
//         module.addSerializer(GrantedAuthority.class, new GrantedAuthoritySerializer());
//         module.addDeserializer(GrantedAuthority.class, new GrantedAuthorityDeserializer());
//         objectMapper.registerModule(module);

//         //objectMapper.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);
//         objectMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
//         objectMapper.setVisibility(PropertyAccessor.FIELD, Visibility.ANY);
//         return objectMapper;
//     }
// }
