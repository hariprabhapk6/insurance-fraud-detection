export default function SectionHeading({ title, children, as: Tag = 'h2', id }) {
  return (
    <div className="section-head">
      <Tag id={id}>{title}</Tag>
      {children && <p>{children}</p>}
    </div>
  );
}
