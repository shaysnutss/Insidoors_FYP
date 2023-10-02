import "./Ticket.css";

export const TicketHighPriority = () => {
  return (
    <div className="ticket-high-priority">
      <img className="line" alt="Line" src="line-8.svg" />
      <div className="overlap">
        <div className="overlap-group">
          <div className="rectangle" />
          <div className="text-wrapper">High</div>
        </div>
        <div className="div">
          <div className="text-wrapper-2">DESCRIPTION</div>
          <div className="text-wrapper-3">Title</div>
          <p className="p">Lorem ipsum dolor sit amet. Consectetur adipiscing elit, sed do eiusmod tempor.</p>
        </div>
      </div>
      <img className="image" alt="Image" src="image-7.png" />
      <img className="img" alt="Image" src="image-8.png" />
    </div>
  );
};