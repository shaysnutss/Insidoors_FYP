package com.service.notificationservice.Email;

import java.io.IOException;

import com.amazonaws.regions.Regions;
import com.amazonaws.services.simpleemail.AmazonSimpleEmailService;
import com.amazonaws.services.simpleemail.AmazonSimpleEmailServiceClientBuilder;
import com.amazonaws.services.simpleemail.model.Body;
import com.amazonaws.services.simpleemail.model.Content;
import com.amazonaws.services.simpleemail.model.Destination;
import com.amazonaws.services.simpleemail.model.Message;
import com.amazonaws.services.simpleemail.model.SendEmailRequest;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;
import org.thymeleaf.TemplateEngine;
import org.thymeleaf.context.Context;

@Component
public class AmazonSES {

    private final TemplateEngine templateEngine;

    public AmazonSES(TemplateEngine templateEngine) {
        this.templateEngine = templateEngine;
    }

    // Official Insidoor's email that is verified with Amazon SES.
    static final String FROM = "insidoorsfyp@gmail.com";


    // The email body for recipients with non-HTML email clients.
    static final String TEXTBODY = "Please check the task management system for the new case that is assigned to you.";


    public String sendEmail(String name, String recipientEmail, String subject, int severity, String incidentTitle, int taskId) throws IOException {

        try {
            AmazonSimpleEmailService client =
                    AmazonSimpleEmailServiceClientBuilder.standard()
                            .withRegion(Regions.AP_SOUTHEAST_2).build();
            SendEmailRequest request = new SendEmailRequest()
                    .withDestination(
                            new Destination().withToAddresses(recipientEmail))
                    .withMessage(new Message()
                            .withBody(new Body()
                                    .withHtml(new Content()
                                            .withCharset("UTF-8").withData(setContent(name, severity, incidentTitle, taskId)))
                                    .withText(new Content()
                                            .withCharset("UTF-8").withData(TEXTBODY)))
                            .withSubject(new Content()
                                    .withCharset("UTF-8").withData(subject)))
                    .withSource(FROM);

            client.sendEmail(request);
            System.out.println("Email sent!");
        } catch (Exception ex) {
            System.out.println("The email was not sent. Error message: "
                    + ex.getMessage());
            return null;
        }
        return "Email sent";
    }

    public String setContent(String name, int severity, String incidentTitle, int taskId){
        // Create a context object and set the input variables in the context object.
        Context context = new Context();
        context.setVariable("name", name);
        context.setVariable("severity", severity);
        context.setVariable("incidentTitle", incidentTitle);
        context.setVariable("taskId", taskId);

//        ClassPathResource image = new ClassPathResource("icon.png");
//        context.setVariable("imageCid", "image001");
//        System.out.println("Image path: " + image.getPath());
        // Process the HTML file using the template engine and the context object.
        String htmlContent = templateEngine.process("email-template", context);
        return htmlContent;
    }
}


